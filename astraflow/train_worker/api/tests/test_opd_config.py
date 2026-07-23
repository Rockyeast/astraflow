import pytest

from astraflow.train_worker.api.cli_args import PPOActorConfig
from astraflow.train_worker.api.opd_config import OPDConfig


def test_opd_config_enforces_algorithm_invariants():
    config = OPDConfig(
        opd_kl_coef=0.7,
        opd_discount=0.25,
        actor=PPOActorConfig(
            kl_ctl=99,
            kl_estimator="k3",
            kl_penalty_coef=3,
            filter_zero_adv_in_batch=True,
        ),
    )

    assert config.actor.kl_ctl == 0.7
    assert config.actor.kl_estimator == "k1"
    assert config.actor.kl_penalty_coef == 0
    assert config.actor.discount == 0.25
    assert config.actor.gae_lambda == 1
    assert config.actor.filter_zero_adv_in_batch is False
    assert config.actor.reward_norm is None
    assert config.actor.adv_norm is None
    assert config.critic is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("opd_kl_coef", 0), ("opd_discount", -0.1), ("opd_discount", 1.1)],
)
def test_opd_config_rejects_invalid_values(field_name, value):
    with pytest.raises(ValueError):
        OPDConfig(**{field_name: value})


def test_opd_zero_reward_has_no_task_signal():
    from astraflow.core.workflow.reward.opd_zero import opd_zero_reward_fn

    assert opd_zero_reward_fn("prompt", "completion") == 0.0
