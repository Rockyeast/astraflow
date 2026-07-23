from datasets import Dataset

from astraflow.train_worker.api.opd_config import OPDConfig

from .ppo_trainer import AstraFlowPPOTrainer


class AstraFlowOPDTrainer(AstraFlowPPOTrainer):
    """OPD trainer reusing AstraFlow's rollout, update, and weight-sync path."""

    def __init__(
        self,
        config: OPDConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | dict[str, tuple[Dataset, int]] | None = None,
    ):
        if config.ref is None:
            raise ValueError("OPD requires ref.path to point to a frozen teacher")
        if config.ref.optimizer is not None:
            raise ValueError("OPD teacher must be frozen (ref.optimizer must be null)")
        super().__init__(config, train_dataset, valid_dataset)
