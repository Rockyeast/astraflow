from __future__ import annotations

import torch


def align_and_compute_opd_advantages(
    rollout_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    coefficient: float,
    discount: float = 0.0,
    recomputed_logprobs: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Align rollout tensors with next-token training positions and score them.

    SGLang stores each sampled logprob on the generated token position.
    AstraFlow's training forward stores the probability of token ``t+1`` on
    position ``t``. The rollout logprobs and response mask therefore move one
    position left. Teacher logprobs already come from the training forward and
    remain unchanged.
    """
    aligned_mask = torch.roll(loss_mask.float(), shifts=-1, dims=-1)
    sampled_logprobs = (
        recomputed_logprobs
        if recomputed_logprobs is not None
        else torch.roll(rollout_logprobs, shifts=-1, dims=-1)
    )
    advantages, reverse_kl, token_rewards = (
        compute_reverse_kl_advantages(
            sampled_logprobs,
            teacher_logprobs,
            aligned_mask,
            coefficient=coefficient,
            discount=discount,
        )
    )
    return (
        sampled_logprobs,
        aligned_mask,
        advantages,
        reverse_kl,
        token_rewards,
    )


def compute_reverse_kl_advantages(
    sampled_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    coefficient: float,
    discount: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-token OPD advantages on tokens sampled by the student.

    This is the sampled reverse-KL estimator used by on-policy distillation:
    ``log p_student(token) - log p_teacher(token)``. Prompt and padding tokens
    are excluded by ``loss_mask``.

    Returns ``(advantages, reverse_kl, token_rewards)``. With zero discount,
    advantages and token rewards are identical. A positive discount lets later
    teacher feedback influence earlier response tokens.
    """
    if sampled_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "student and teacher logprobs must have identical shapes, got "
            f"{tuple(sampled_logprobs.shape)} and {tuple(teacher_logprobs.shape)}"
        )
    if sampled_logprobs.shape != loss_mask.shape:
        raise ValueError(
            "logprobs and loss_mask must have identical shapes, got "
            f"{tuple(sampled_logprobs.shape)} and {tuple(loss_mask.shape)}"
        )
    if coefficient <= 0:
        raise ValueError("coefficient must be positive")
    if not 0 <= discount <= 1:
        raise ValueError("discount must be between 0 and 1")

    mask = loss_mask.to(device=sampled_logprobs.device, dtype=torch.float32)
    reverse_kl = (
        sampled_logprobs.float() - teacher_logprobs.float()
    ) * mask
    token_rewards = -coefficient * reverse_kl

    if discount == 0:
        return token_rewards, reverse_kl, token_rewards

    advantages = torch.empty_like(token_rewards)
    running = torch.zeros_like(token_rewards[..., -1])
    for token_idx in range(token_rewards.shape[-1] - 1, -1, -1):
        running = token_rewards[..., token_idx] + discount * running
        advantages[..., token_idx] = running

    advantages = advantages * mask
    return advantages, reverse_kl, token_rewards
