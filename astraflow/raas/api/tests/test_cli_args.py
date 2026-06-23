import json
import sys
import types

ray = types.ModuleType("ray")
ray.remote = lambda obj=None, **_kwargs: obj if obj is not None else (lambda x: x)
sys.modules.setdefault("ray", ray)
transformers = types.ModuleType("transformers")
transformers_utils = types.ModuleType("transformers.utils")
transformers_import_utils = types.ModuleType("transformers.utils.import_utils")
transformers_import_utils.is_torch_npu_available = lambda: False
transformers_utils.import_utils = transformers_import_utils
transformers.utils = transformers_utils
sys.modules.setdefault("transformers", transformers)
sys.modules.setdefault("transformers.utils", transformers_utils)
sys.modules.setdefault("transformers.utils.import_utils", transformers_import_utils)

from astraflow.raas.api import cli_args
from astraflow.raas.api.cli_args import SGLangConfig


class _FakePlatform:
    device_type = "cuda"


def test_sglang_config_serializes_vortex_dict(monkeypatch):
    monkeypatch.setattr(
        "astraflow.raas.platforms.current_platform",
        _FakePlatform(),
        raising=False,
    )
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(cli_args, "is_version_less", lambda *_args, **_kwargs: False)

    args = SGLangConfig.build_args(
        SGLangConfig(
            model_path="Qwen/Qwen3-0.6B",
            context_length=512,
            page_size=16,
            vortex={
                "module_name": "gqa_block_sparse_attention",
                "topk_val": 30,
                "block_size": 16,
            },
        ),
        tp_size=1,
        base_gpu_id=0,
        host="127.0.0.1",
        port=30000,
    )

    assert "vortex" not in args
    assert args["page_size"] == 16
    assert json.loads(args["vortex_config"]) == {
        "module_name": "gqa_block_sparse_attention",
        "topk_val": 30,
        "block_size": 16,
    }

    cmd = cli_args.get_py_cmd("astraflow.raas.entrypoint", args)
    assert "--vortex-config" in cmd
    assert "--vortex" not in cmd


def test_sglang_config_preserves_vortex_json_string(monkeypatch):
    monkeypatch.setattr(
        "astraflow.raas.platforms.current_platform",
        _FakePlatform(),
        raising=False,
    )
    monkeypatch.setattr(
        cli_args.pkg_version,
        "is_version_greater_or_equal",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(cli_args, "is_version_less", lambda *_args, **_kwargs: False)

    vortex_json = '{"module_name":"gqa_block_sparse_attention","topk_val":30}'
    args = SGLangConfig.build_args(
        SGLangConfig(model_path="Qwen/Qwen3-0.6B", vortex=vortex_json),
        tp_size=1,
        base_gpu_id=0,
    )

    assert args["vortex_config"] == vortex_json
