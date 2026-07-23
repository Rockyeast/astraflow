from .ppo_base import PPOTrainerBase
from .opd_trainer import AstraFlowOPDTrainer
from .ppo_trainer import AstraFlowPPOTrainer

# Backward-compatible alias for examples/ that still reference the old name.
PPOTrainer = AstraFlowPPOTrainer

__all__ = [
    "AstraFlowPPOTrainer",
    "AstraFlowOPDTrainer",
    "PPOTrainer",
    "PPOTrainerBase",
]
