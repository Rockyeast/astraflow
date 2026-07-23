from dataclasses import dataclass, field

from .cli_args import PPOConfig


@dataclass
class OPDConfig(PPOConfig):
    """On-policy distillation recipe built on AstraFlow's PPO infrastructure."""

    opd_kl_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the per-token teacher KL signal."},
    )
    opd_discount: float = field(
        default=0.0,
        metadata={"help": "Discount applied to future per-token teacher KL."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.opd_kl_coef <= 0:
            raise ValueError("opd_kl_coef must be positive")
        if not 0 <= self.opd_discount <= 1:
            raise ValueError("opd_discount must be between 0 and 1")

        # OPD uses the rollout policy versus a frozen teacher as its dense,
        # per-token signal. Keep these invariants out of user-authored PPO knobs.
        self.actor.kl_ctl = self.opd_kl_coef
        self.actor.kl_estimator = "k1"
        self.actor.kl_penalty_coef = 0.0
        self.actor.discount = self.opd_discount
        self.actor.gae_lambda = 1.0
        self.actor.reward_norm = None
        self.actor.adv_norm = None
        self.actor.filter_zero_adv_in_batch = False
        self.actor.eps_clip = 100.0
        self.actor.eps_clip_higher = 100.0
        self.actor.m2_threshold = None
        self.actor.use_sapo_loss = False
        self.critic = None
