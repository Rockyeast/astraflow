from pathlib import Path

from omegaconf import OmegaConf

from astraflow.core.config.loader import (
    load_and_merge_configs,
    load_dataflow_config,
    load_raas_config,
    load_trainer_config,
)
from astraflow.dataflow.dataset.opd_smoke import get_opd_smoke_dataset
from astraflow.train_worker.api.cli_args import to_structured_cfg
from astraflow.train_worker.api.opd_config import OPDConfig


def _recipe_dir() -> Path:
    root = Path(__file__).resolve().parents[4]
    return root / "examples/math/qwen3-1.7b-opd-smoke/yaml"


def test_opd_smoke_recipe_resolves_student_teacher_and_algorithm():
    raw = load_and_merge_configs([str(_recipe_dir() / "experiment.yaml")])
    trainer = load_trainer_config(raw, trainer_key="trainer_model0")
    config = OmegaConf.to_object(
        to_structured_cfg(OmegaConf.create(trainer), OPDConfig)
    )

    assert isinstance(config, OPDConfig)
    assert config.actor.path == "Qwen/Qwen3-1.7B"
    assert config.ref is not None
    assert config.ref.path == "Qwen/Qwen3-4B"
    assert config.ref.optimizer is None
    assert config.total_train_steps == 2
    assert config.train_batch_size == 2
    assert config.actor.kl_ctl == 1.0
    assert config.actor.discount == 0.0
    assert config.sync_weight_updates is True


def test_opd_smoke_recipe_uses_local_prompts_and_one_rollout_engine():
    config_dir = _recipe_dir()
    raw = load_and_merge_configs(
        [str(config_dir / "experiment.yaml"), str(config_dir / "raas.yaml")]
    )

    dataflow = load_dataflow_config(raw)
    raas = load_raas_config(raw)

    assert (
        dataflow["agent"]["rollout_dataset"]["dataset_fn"]
        == "astraflow.dataflow.dataset.opd_smoke:get_opd_smoke_dataset"
    )
    assert dataflow["agent"]["filter_function"] == "keep_all"
    assert dataflow["agent"]["max_staleness"] == 0
    assert dataflow["agent"]["workflow_spec"]["reward_fn"] == "opd_zero"
    assert raas["models"]["model0"]["sglang"]["model_path"] == "Qwen/Qwen3-1.7B"
    assert raas["allocation_mode"]["model0"]["data_parallel_size"] == 1


def test_opd_smoke_dataset_is_local_and_deterministic():
    dataset = get_opd_smoke_dataset()

    assert len(dataset) == 2
    assert dataset["query_id"] == [
        "opd_smoke-00000000",
        "opd_smoke-00000001",
    ]
    assert all(row["source"] == "opd_smoke" for row in dataset)
