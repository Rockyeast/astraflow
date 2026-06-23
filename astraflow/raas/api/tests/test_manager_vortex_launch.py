import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ray = types.ModuleType("ray")
ray.remote = lambda obj=None, **_kwargs: obj if obj is not None else (lambda x: x)
sys.modules.setdefault("ray", ray)

transformers = types.ModuleType("transformers")
transformers.PreTrainedTokenizerFast = type("PreTrainedTokenizerFast", (), {})
transformers_utils = types.ModuleType("transformers.utils")
transformers_import_utils = types.ModuleType("transformers.utils.import_utils")
transformers_import_utils.is_torch_npu_available = lambda: False
transformers_utils.import_utils = transformers_import_utils
transformers.utils = transformers_utils
sys.modules.setdefault("transformers", transformers)
sys.modules.setdefault("transformers.utils", transformers_utils)
sys.modules.setdefault("transformers.utils.import_utils", transformers_import_utils)

engine_pkg = types.ModuleType("astraflow.raas.engine")
engine_pkg.__path__ = []
sglang_remote = types.ModuleType("astraflow.raas.engine.sglang_remote")
vllm_remote = types.ModuleType("astraflow.raas.engine.vllm_remote")
sglang_remote.SGLangEngine = type("SGLangEngine", (), {})
vllm_remote.VLLMEngine = type("VLLMEngine", (), {})
sys.modules.setdefault("astraflow.raas.engine", engine_pkg)
sys.modules.setdefault("astraflow.raas.engine.sglang_remote", sglang_remote)
sys.modules.setdefault("astraflow.raas.engine.vllm_remote", vllm_remote)

workflow_pkg = types.ModuleType("astraflow.core.workflow")
workflow_api_pkg = types.ModuleType("astraflow.core.workflow.api")
workflow_engine_api = types.ModuleType("astraflow.core.workflow.api.engine_api")
workflow_registry = types.ModuleType("astraflow.core.workflow.registry")
workflow_engine_api.EngineGroup = type("EngineGroup", (), {})
workflow_registry.get_reward = lambda *_args, **_kwargs: None
workflow_registry.get_workflow = lambda *_args, **_kwargs: None
sys.modules.setdefault("astraflow.core.workflow", workflow_pkg)
sys.modules.setdefault("astraflow.core.workflow.api", workflow_api_pkg)
sys.modules.setdefault("astraflow.core.workflow.api.engine_api", workflow_engine_api)
sys.modules.setdefault("astraflow.core.workflow.registry", workflow_registry)

from astraflow.raas.api.alloc_mode import AllocationMode


_MANAGER_PATH = Path(__file__).resolve().parents[2] / "server" / "manager.py"
_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "_astraflow_raas_manager_under_test",
    _MANAGER_PATH,
)
manager_module = importlib.util.module_from_spec(_MANAGER_SPEC)
assert _MANAGER_SPEC and _MANAGER_SPEC.loader
_MANAGER_SPEC.loader.exec_module(manager_module)
RaaS3Manager = manager_module.RaaS3Manager


class _FakeEngine:
    def __init__(self):
        self.launched_args = []

    def launch_server(self, server_args):
        self.launched_args.append(server_args)


def test_vortex_config_forces_autopatch(monkeypatch):
    manager = RaaS3Manager(service_port=19190)
    engine = _FakeEngine()

    monkeypatch.setattr(
        manager,
        "_build_server_args",
        lambda **_kwargs: {"vortex_config": '{"topk_val":30}'},
    )
    monkeypatch.setattr(
        manager,
        "_extract_visible_devices",
        lambda _total_fallback: ["0"],
    )
    monkeypatch.setattr(
        manager_module,
        "find_free_ports",
        lambda _n, exclude_ports=None: [30000],
    )
    monkeypatch.setattr(manager_module, "gethostip", lambda: "127.0.0.1")

    config = SimpleNamespace(
        cluster=SimpleNamespace(n_gpus_per_node=1),
        weight_transfer_mode="disk",
    )
    allocation_mode = AllocationMode.from_engine_config(
        {"backend": "sglang", "data_parallel_size": 1, "tensor_parallel_size": 1}
    )

    manager._launch_inference_servers(
        engine=engine,
        backend="sglang",
        config=config,
        allocation_mode=allocation_mode,
    )

    assert len(engine.launched_args) == 1
    launch_env = engine.launched_args[0]["__launch_env__"]
    assert launch_env["ASTRAFLOW_AUTOPATCH"] == "true"
    assert engine.launched_args[0]["vortex_config"] == '{"topk_val":30}'
