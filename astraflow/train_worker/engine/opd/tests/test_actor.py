from types import SimpleNamespace

import pytest
import torch

from astraflow.train_worker.api.cli_args import PPOActorConfig
from astraflow.train_worker.engine.opd.actor import OPDActor
def _actor(**overrides) -> OPDActor:
    config = PPOActorConfig(
        kl_ctl=overrides.pop("kl_ctl", 1.0),
        kl_estimator="k1",
        discount=overrides.pop("discount", 0.0),
        reward_norm=None,
        adv_norm=None,
        **overrides,
    )
    return OPDActor(config, SimpleNamespace())


def test_opd_aligns_next_token_logprobs_and_masks_prompt_tokens():
    actor = _actor()
    batch = {
        "logprobs": torch.tensor([[99.0, 99.0, -2.0, -3.0, -4.0]]),
        "ref_logp": torch.tensor([[0.0, -1.5, -2.0, -4.5, 0.0]]),
        "loss_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]),
    }

    result = actor.compute_opd_advantages(batch)

    torch.testing.assert_close(
        result["loss_mask"],
        torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        result["opd_reverse_kl"],
        torch.tensor([[0.0, -0.5, -1.0, 0.5, 0.0]]),
    )
    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.0, 0.5, 1.0, -0.5, 0.0]]),
    )
def test_opd_requires_teacher_logprobs():
    actor = _actor()
    with pytest.raises(ValueError, match="ref_logp"):
        actor.compute_opd_advantages(
            {
                "logprobs": torch.zeros(1, 2),
                "loss_mask": torch.ones(1, 2),
            }
        )
