"""AstraFlow on-policy distillation training entrypoint."""

import sys

from datasets import Dataset

from astraflow.train_worker.api.cli_args import load_expr_config
from astraflow.train_worker.api.opd_config import OPDConfig
from astraflow.train_worker.trainer.opd_trainer import AstraFlowOPDTrainer


def main(args):
    config, _ = load_expr_config(args, OPDConfig)
    n_dummy = config.train_batch_size * min(config.total_train_epochs, 100)
    train_dataset = Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "dummy"}]] * n_dummy}
    )

    with AstraFlowOPDTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
