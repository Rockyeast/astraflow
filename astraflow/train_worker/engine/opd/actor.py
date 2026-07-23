from __future__ import annotations

from typing import Any

import torch

from astraflow.train_worker.engine.opd.functional import (
    align_and_compute_opd_advantages,
)
from astraflow.train_worker.engine.ppo.actor import PPOActor
from astraflow.train_worker.utils.perf_tracer import trace_perf


class OPDActor(PPOActor):
    """PPO update machinery with an explicit teacher reverse-KL signal."""

    @trace_perf("opd_actor.compute_advantages", category="compute")
    @torch.no_grad()
    def compute_opd_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        teacher_logp = data.get("ref_logp")
        if teacher_logp is None:
            raise ValueError(
                "OPD requires ref_logp from the frozen teacher for every batch"
            )

        recomputed_logp = None
        if self.config.recompute_logprob:
            recomputed_logp = data.get("prox_logp")
            if recomputed_logp is None:
                raise ValueError(
                    "prox_logp is required when recompute_logprob=True"
                )

        (
            sampled_logp,
            loss_mask,
            advantages,
            reverse_kl,
            token_rewards,
        ) = align_and_compute_opd_advantages(
            data["logprobs"],
            teacher_logp,
            data["loss_mask"],
            coefficient=self.kl_ctl,
            discount=self.discount,
            recomputed_logprobs=recomputed_logp,
        )

        data["advantages"] = advantages
        data["returns"] = advantages
        data["kl_rewards"] = token_rewards
        data["tot_rewards"] = token_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = sampled_logp * loss_mask
        data["opd_reverse_kl"] = reverse_kl
        return data

    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.compute_opd_advantages(data)

    def compute_advantages_with_normalized_reward(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        return self.compute_opd_advantages(data)

    def ppo_update(self, data: dict[str, Any]) -> None:
        # This diagnostic tensor is not a model input. The inherited PPO
        # updater already logs the equivalent KL reward before training.
        data.pop("opd_reverse_kl", None)
        super().ppo_update(data)
