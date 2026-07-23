from astraflow.core.config.loader import load_raas_config


def test_load_raas_config_preserves_vortex_sglang_precedence():
    raw = {
        "experiment": {
            "model_path": "Qwen/Qwen3-0.6B",
            "tokenizer_path": "Qwen/Qwen3-0.6B",
            "seed": 7,
            "dtype": "bfloat16",
        },
        "raas": {
            "models": {
                "model0": {
                    "sglang": {
                        "vortex": {
                            "topk_val": 31,
                        }
                    }
                }
            },
            "sglang": {
                "vortex": {
                    "module_name": "base_sparse_attention",
                    "topk_val": 8,
                    "block_size": 32,
                }
            },
        },
        "sglang": {
            "context_length": 512,
            "vortex": {
                "module_name": "gqa_block_sparse_attention",
                "block_size": 16,
            },
        },
    }

    config = load_raas_config(raw)

    assert config["sglang"]["vortex"] == {
        "module_name": "gqa_block_sparse_attention",
        "topk_val": 8,
        "block_size": 16,
    }
    assert config["models"]["model0"]["sglang"]["vortex"] == {
        "module_name": "gqa_block_sparse_attention",
        "topk_val": 31,
        "block_size": 16,
    }
    assert config["models"]["model0"]["sglang"]["model_path"] == "Qwen/Qwen3-0.6B"
    assert config["models"]["model0"]["sglang"]["random_seed"] == 7


def test_example_raas_vortex_yaml_loads():
    from pathlib import Path

    from astraflow.core.config.loader import load_and_merge_configs

    root = Path(__file__).resolve().parents[4]
    config_dir = root / "examples/math/qwen3-1.7b-m2po-delta/yaml"
    raw = load_and_merge_configs(
        [
            str(config_dir / "experiment.yaml"),
            str(config_dir / "raas_vortex.yaml"),
        ]
    )

    config = load_raas_config(raw)
    vortex = config["models"]["model0"]["sglang"]["vortex"]

    assert vortex["module_name"] == "gqa_block_sparse_attention"
    assert vortex["topk_val"] == 30
    assert vortex["block_size"] == 16
    assert config["models"]["model0"]["sglang"]["page_size"] == 16
    assert config["models"]["model0"]["sglang"]["attention_backend"] == "flashinfer"
    assert config["models"]["model0"]["sglang"]["model_path"] == "Qwen/Qwen3-1.7B"
