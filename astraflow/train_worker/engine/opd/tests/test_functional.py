import pytest
import torch

from astraflow.train_worker.engine.opd.functional import (
    align_and_compute_opd_advantages,
    compute_reverse_kl_advantages,
)
from astraflow.train_worker.utils.functional import ppo_actor_loss_fn


def test_alignment_moves_rollout_values_but_not_teacher_values():
    (
        sampled_logprobs,
        loss_mask,
        advantages,
        reverse_kl,
        _,
    ) = align_and_compute_opd_advantages(
        rollout_logprobs=torch.tensor(
            [[99.0, 99.0, -2.0, -3.0, -4.0]]
        ),
        teacher_logprobs=torch.tensor(
            [[0.0, -1.5, -2.0, -4.5, 0.0]]
        ),
        loss_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]),
        coefficient=1.0,
    )

    torch.testing.assert_close(
        sampled_logprobs,
        torch.tensor([[99.0, -2.0, -3.0, -4.0, 99.0]]),
    )
    torch.testing.assert_close(
        loss_mask,
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        reverse_kl,
        torch.tensor([[0.0, -0.5, -1.0, 0.5, 0.0]]),
    )
    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.0, 0.5, 1.0, -0.5, 0.0]]),
    )


def test_recomputed_logprobs_are_already_training_aligned():
    recomputed = torch.tensor([[8.0, -2.0, -3.0, -4.0, 8.0]])
    sampled_logprobs, *_ = align_and_compute_opd_advantages(
        rollout_logprobs=torch.full((1, 5), 99.0),
        teacher_logprobs=torch.zeros(1, 5),
        loss_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]),
        coefficient=1.0,
        recomputed_logprobs=recomputed,
    )

    torch.testing.assert_close(sampled_logprobs, recomputed)


def test_reverse_kl_masks_prompt_and_padding_tokens():
    advantages, reverse_kl, rewards = compute_reverse_kl_advantages(
        sampled_logprobs=torch.tensor([[8.0, -2.0, -3.0, -4.0, 8.0]]),
        teacher_logprobs=torch.tensor([[0.0, -1.5, -2.0, -4.5, 0.0]]),
        loss_mask=torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]]),
        coefficient=1.0,
    )

    torch.testing.assert_close(
        reverse_kl,
        torch.tensor([[0.0, -0.5, -1.0, 0.5, 0.0]]),
    )
    torch.testing.assert_close(
        rewards,
        torch.tensor([[-0.0, 0.5, 1.0, -0.5, -0.0]]),
    )
    torch.testing.assert_close(advantages, rewards)


def test_discount_propagates_future_teacher_signal():
    advantages, reverse_kl, rewards = compute_reverse_kl_advantages(
        sampled_logprobs=torch.tensor([[0.0, 2.0, 1.0]]),
        teacher_logprobs=torch.zeros(1, 3),
        loss_mask=torch.ones(1, 3),
        coefficient=1.0,
        discount=0.5,
    )

    torch.testing.assert_close(reverse_kl, torch.tensor([[0.0, 2.0, 1.0]]))
    torch.testing.assert_close(rewards, torch.tensor([[-0.0, -2.0, -1.0]]))
    torch.testing.assert_close(
        advantages,
        torch.tensor([[-1.25, -2.5, -1.0]]),
    )


def test_ppo_update_moves_student_logprobs_toward_teacher_signal():
    old_logprobs = torch.tensor([[-2.0, -2.0]])
    teacher_logprobs = torch.tensor([[-1.0, -3.0]])
    loss_mask = torch.ones(1, 2, dtype=torch.bool)
    advantages, _, _ = compute_reverse_kl_advantages(
        old_logprobs,
        teacher_logprobs,
        loss_mask,
        coefficient=1.0,
    )
    new_logprobs = old_logprobs.clone().requires_grad_(True)

    loss, _ = ppo_actor_loss_fn(
        logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=100.0,
        loss_mask=loss_mask,
        eps_clip_higher=100.0,
    )
    loss.backward()

    assert new_logprobs.grad is not None
    assert new_logprobs.grad[0, 0] < 0
    assert new_logprobs.grad[0, 1] > 0


@pytest.mark.parametrize(
    ("coefficient", "discount"),
    [(0.0, 0.0), (1.0, -0.1), (1.0, 1.1)],
)
def test_rejects_invalid_hyperparameters(coefficient, discount):
    with pytest.raises(ValueError):
        compute_reverse_kl_advantages(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            coefficient=coefficient,
            discount=discount,
        )
