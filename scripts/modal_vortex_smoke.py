from __future__ import annotations

import json
import os

import modal


APP_NAME = "astraflow-vortex-smoke"
VORTEX_REF = os.environ.get("VORTEX_TORCH_REF", "v0.6")
HF_CACHE_DIR = "/root/.cache/huggingface"


def _modal_gpu_spec(value: str) -> str | list[str]:
    name, sep, count = value.partition(":")
    if sep and count.isdigit():
        return [name] * int(count)
    return value


GPU_TYPE_RAW = os.environ.get("VORTEX_MODAL_GPU", "L40S")
GPU_TYPE = _modal_gpu_spec(GPU_TYPE_RAW)
FULL_CHAIN_GPU_TYPE = _modal_gpu_spec(os.environ.get(
    "VORTEX_MODAL_FULL_CHAIN_GPU",
    GPU_TYPE_RAW if ":" in GPU_TYPE_RAW else f"{GPU_TYPE_RAW}:2",
))

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("vortex-hf-cache", create_if_missing=True)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "curl",
        "git",
        "libnuma1",
        "libnuma-dev",
        "ninja-build",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_CUDA_ARCH_LIST": "8.9;9.0",
        }
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        f"git clone -b {VORTEX_REF} --recursive https://github.com/Infini-AI-Lab/vortex_torch.git /workspace/vortex_torch",
        'cd /workspace/vortex_torch/third_party/sglang/v0.5.9/sglang && python -m pip install -e "python"',
        "cd /workspace/vortex_torch && python -m pip install -e .",
    )
)

astraflow_image = (
    image.add_local_dir("src", "/workspace/astraflow", copy=True)
    .run_commands(
        "python -m pip install --no-deps -e /workspace/astraflow",
        'python -m pip install aiofiles aiohttp colorama colorlog datasets fastapi flask math-verify omegaconf orjson psutil pybase64 PyYAML "ray[default]" requests rich torchdata tqdm uvicorn uvloop',
        "python -m pip install einops msgspec nvidia-ml-py peft sentencepiece setproctitle tabulate tensorboardx torch_memory_saver wandb",
        'python -m pip install blosc func_timeout json5 mathruler==0.1.0 mbridge==0.13.0 megatron-core==0.13.1 numba pebble prettytable pylatexenc==2.10 python-dotenv redis "swanlab[dashboard]==0.6.12" swanboard==0.1.9b1 timeout-decorator word2number',
        "python -m pip install numpy==2.4.0",
    )
    .env({"PYTHONPATH": "/workspace/astraflow"})
)


def _vortex_json() -> str:
    return json.dumps(
        {
            "topk_val": 30,
            "block_size": 16,
            "module_name": "gqa_block_sparse_attention",
            "attention_backend": "flashinfer",
            "impl_backend": "triton",
            "workload_chunk_size": 32,
            "dtype": "bfloat16",
            "layers_skip": [0],
            "block_reserved_bos": 1,
            "block_reserved_eos": 1,
            "max_seq_lens": 512,
            "max_topk_val": 64,
            "compilation_cache_dir": "/tmp/vortex_compilation_cache",
        },
        separators=(",", ":"),
    )


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 60)
def parse_smoke() -> str:
    import sglang  # noqa: F401
    import vortex_torch  # noqa: F401  # installs ServerArgs adapter
    from sglang.srt.server_args import prepare_server_args
    from vortex_torch.engine.sgl.config import VortexConfig

    vortex_json = _vortex_json()
    server_args = prepare_server_args(
        [
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "16",
            "--tp-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--vortex-config",
            vortex_json,
        ]
    )

    result = {
        "sglang_file": getattr(sglang, "__file__", None),
        "vortex_file": getattr(vortex_torch, "__file__", None),
        "vortex_type": type(server_args.vortex).__name__,
        "is_vortex_config": isinstance(server_args.vortex, VortexConfig),
        "enable_vortex_sparsity": server_args.enable_vortex_sparsity,
        "vortex_topk_val": server_args.vortex_topk_val,
        "vortex_block_size": server_args.vortex_block_size,
        "vortex_module_name": server_args.vortex_module_name,
        "vortex_attention_backend": server_args.vortex_attention_backend,
    }

    assert result["is_vortex_config"], result
    assert result["enable_vortex_sparsity"] is True, result
    assert result["vortex_topk_val"] == 30, result
    assert result["vortex_block_size"] == 16, result
    return json.dumps(result, indent=2, sort_keys=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
)
def server_smoke(model_path: str = "Qwen/Qwen3-0.6B") -> str:
    import json as _json
    import os
    import signal
    import subprocess
    import time
    from collections import deque

    import requests
    import torch

    host = "127.0.0.1"
    port = 30000
    base_url = f"http://{host}:{port}"
    logs: deque[str] = deque(maxlen=220)
    vortex_json = _vortex_json()

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    cmd = [
        "python",
        "-c",
        (
            "import os, sys\n"
            "import vortex_torch\n"
            "from sglang.launch_server import run_server\n"
            "from sglang.srt.server_args import prepare_server_args\n"
            "from sglang.srt.utils import kill_process_tree\n"
            "server_args = prepare_server_args(sys.argv[1:])\n"
            "try:\n"
            "    run_server(server_args)\n"
            "finally:\n"
            "    kill_process_tree(os.getpid(), include_parent=False)\n"
        ),
        "--model-path",
        model_path,
        "--page-size",
        "16",
        "--attention-backend",
        "flashinfer",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--context-length",
        "512",
        "--mem-fraction-static",
        "0.65",
        "--max-running-requests",
        "4",
        "--tp-size",
        "1",
        "--host",
        host,
        "--port",
        str(port),
        "--vortex-config",
        vortex_json,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    def drain_logs() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            logs.append(line.rstrip())

    import threading

    thread = threading.Thread(target=drain_logs, daemon=True)
    thread.start()

    try:
        ready = False
        deadline = time.time() + 20 * 60
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "SGLang server exited before becoming healthy.\n"
                    + "\n".join(logs)
                )
            try:
                health = requests.get(f"{base_url}/health", timeout=5)
                if health.status_code == 200:
                    ready = True
                    break
                last_error = f"health status={health.status_code} body={health.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
            time.sleep(5)

        if not ready:
            raise TimeoutError(
                f"SGLang server did not become healthy: {last_error}\n"
                + "\n".join(logs)
            )

        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": "Hello, my name is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                },
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        generate_json = response.json()

        result = {
            "model_path": model_path,
            "gpu": smi.stdout.strip(),
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "server_ready": ready,
            "generate_response": generate_json,
            "vortex_config": _json.loads(vortex_json),
            "command": cmd,
            "log_tail": list(logs)[-80:],
        }
        return _json.dumps(result, indent=2, sort_keys=True)
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=30)


@app.function(
    image=astraflow_image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
)
def raas_manager_smoke(model_path: str = "Qwen/Qwen3-0.6B") -> str:
    import importlib.util
    import json as _json
    import os
    import sys
    import types
    from pathlib import Path
    from types import SimpleNamespace

    import requests
    import torch

    sys.path.insert(0, "/workspace/astraflow")
    astraflow_pkg = types.ModuleType("astraflow")
    astraflow_pkg.__path__ = ["/workspace/astraflow/astraflow"]
    sys.modules["astraflow"] = astraflow_pkg

    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: False
    ray.remote = lambda obj=None, **_kwargs: obj if obj is not None else (lambda x: x)
    sys.modules.setdefault("ray", ray)

    # manager.py imports workflow symbols at module import time. This smoke only
    # exercises the RaaS -> SGLang launch path, so keep workflow deps out of it.
    workflow_pkg = types.ModuleType("astraflow.core.workflow")
    workflow_api_pkg = types.ModuleType("astraflow.core.workflow.api")
    workflow_engine_api = types.ModuleType("astraflow.core.workflow.api.engine_api")
    workflow_registry = types.ModuleType("astraflow.core.workflow.registry")

    class EngineGroup:
        def __init__(self, engines):
            self.engines = engines

    workflow_engine_api.EngineGroup = EngineGroup
    workflow_registry.get_reward = lambda *_args, **_kwargs: None
    workflow_registry.get_workflow = lambda *_args, **_kwargs: None
    sys.modules.setdefault("astraflow.core.workflow", workflow_pkg)
    sys.modules.setdefault("astraflow.core.workflow.api", workflow_api_pkg)
    sys.modules.setdefault("astraflow.core.workflow.api.engine_api", workflow_engine_api)
    sys.modules.setdefault("astraflow.core.workflow.registry", workflow_registry)

    from astraflow.raas.api.alloc_mode import AllocationMode
    from astraflow.raas.api import cli_args as astra_cli_args
    from astraflow.raas.api.cli_args import InferenceEngineConfig, SGLangConfig
    from astraflow.raas.engine.sglang_remote import SGLangEngine

    astra_cli_args.pkg_version.is_version_greater_or_equal = (
        lambda *_args, **_kwargs: True
    )
    astra_cli_args.is_version_less = lambda *_args, **_kwargs: False

    manager_path = Path("/workspace/astraflow/astraflow/raas/server/manager.py")
    spec = importlib.util.spec_from_file_location(
        "_astraflow_raas_manager_for_modal_smoke",
        manager_path,
    )
    manager_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(manager_module)
    RaaS3Manager = manager_module.RaaS3Manager

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rollout_config = InferenceEngineConfig(
        setup_timeout=20 * 60,
        request_timeout=120,
        max_concurrent_rollouts=4,
        consumer_batch_size=4,
    )
    sglang_config = SGLangConfig(
        model_path=model_path,
        context_length=512,
        page_size=16,
        mem_fraction_static=0.65,
        max_running_requests=4,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        vortex=_json.loads(_vortex_json()),
    )
    config = SimpleNamespace(
        rollout=rollout_config,
        sglang=sglang_config,
        cluster=SimpleNamespace(n_gpus_per_node=1),
        weight_transfer_mode="disk",
    )
    allocation_mode = AllocationMode.from_engine_config(
        {"backend": "sglang", "data_parallel_size": 1, "tensor_parallel_size": 1}
    )

    manager = RaaS3Manager(service_port=19190, service_host="127.0.0.1")
    engine = SGLangEngine(rollout_config)
    manager._launch_inference_servers(
        engine=engine,
        backend="sglang",
        config=config,
        allocation_mode=allocation_mode,
    )

    try:
        engine.initialize(engine_id="raas-manager-smoke")
        addr = engine._engine.addresses[0]
        response = requests.post(
            f"http://{addr}/generate",
            json={
                "input_ids": [9707],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = {
            "model_path": model_path,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "addresses": engine._engine.addresses,
            "local_process_count": len(engine._engine.local_server_processes),
            "subprocess_alive": engine._engine.check_subprocesses_alive(),
            "generate_response": response.json(),
            "vortex_config": _json.loads(_vortex_json()),
        }
        return _json.dumps(result, indent=2, sort_keys=True)
    finally:
        engine.destroy()


@app.function(
    image=astraflow_image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=60 * 60,
)
def dataflow_raas_batch_smoke(model_path: str = "Qwen/Qwen3-0.6B") -> str:
    import json as _json
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import threading
    import time
    from collections import deque
    from pathlib import Path

    import requests

    root = Path("/tmp/astraflow_dataflow_raas_smoke")
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))

    dataset_py = root / "smoke_dataset.py"
    dataset_py.write_text(
        textwrap.dedent(
            """
            from datasets import Dataset


            def get_dataset(tokenizer=None, max_length=None, max_samples=None, **kwargs):
                n = max_samples or 8
                rows = {
                    "messages": [
                        [{"role": "user", "content": "What is 1+1? Put the final answer in \\\\boxed{}."}]
                        for _ in range(n)
                    ],
                    "answer": ["\\\\boxed{2}"] * n,
                    "source": ["smoke"] * n,
                    "query_id": [f"smoke-{i}" for i in range(n)],
                }
                return Dataset.from_dict(rows)
            """
        ),
        encoding="utf-8",
    )

    experiment_yaml = root / "experiment.yaml"
    experiment_yaml.write_text(
        textwrap.dedent(
            f"""
            experiment:
              experiment_name: astraflow-smoke
              trial_name: qwen3-0.6b-vortex-dataflow-raas
              fileroot: "{root}/run"
              model_path: "{model_path}"
              tokenizer_path: "{model_path}"
              seed: 1
              dtype: bfloat16
              weight_transfer_mode: tcp
              weight_transfer_strategies: full

            raas:
              models:
                model0:
                  backend: sglang
                  gconfig:
                    n_samples: 1
                    temperature: 0.0
                    max_new_tokens: 8
                    min_new_tokens: 0
              delta_full_sync_interval: 0

            dataflow:
              host: "127.0.0.1"
              port: 8000
              buffer:
                size: 32
                replay_size: 0
                replay_ratio: 0
                max_staleness: 8
                filter_function: keep_all
              rollout_dataset:
                dataset_fn: "smoke_dataset:get_dataset"
                max_length: 128
                max_samples: 8
                batch_size: 1
              workflow_spec:
                workflow_cls: "rlvr"
                reward_fn: "math_verify"
                enable_thinking: false
            """
        ),
        encoding="utf-8",
    )

    raas_yaml = root / "raas.yaml"
    raas_yaml.write_text(
        textwrap.dedent(
            f"""
            rollout:
              max_concurrent_rollouts: 4
              max_concurrent_evals: 1
              pause_grace_period: 1
              enable_adaptive_availability: true
              target_waiting_queue_per_dp: 2
              adaptive_step_size: 1
              load_cache_ttl_ms: 100

            engine:
              model0:
                backend: sglang
                data_parallel_size: 1
                tensor_parallel_size: 1

            sglang:
              context_length: 512
              page_size: 16
              mem_fraction_static: 0.65
              max_running_requests: 4
              skip_tokenizer_init: false
              disable_cuda_graph: true
              disable_overlap_schedule: true
              attention_backend: flashinfer
              vortex:
                module_name: gqa_block_sparse_attention
                topk_val: 30
                block_size: 16
                attention_backend: flashinfer
                impl_backend: triton
                workload_chunk_size: 32
                dtype: bfloat16
                layers_skip:
                  - 0
                block_reserved_bos: 1
                block_reserved_eos: 1
                max_seq_lens: 512
                max_topk_val: 64
                compilation_cache_dir: "{root}/vortex_cache"
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:/workspace/astraflow:" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"

    logs: dict[str, deque[str]] = {
        "dataflow": deque(maxlen=220),
        "raas": deque(maxlen=260),
    }
    procs: dict[str, subprocess.Popen] = {}

    def launch(name: str, cmd: list[str], extra_env: dict[str, str] | None = None):
        proc_env = env.copy()
        if extra_env:
            proc_env.update(extra_env)
        proc = subprocess.Popen(
            cmd,
            cwd="/workspace/astraflow",
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        procs[name] = proc

        def drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logs[name].append(line.rstrip())

        threading.Thread(target=drain, daemon=True).start()
        return proc

    def wait_json(url: str, name: str, timeout_s: float = 900.0):
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            proc = procs.get(name)
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited early with code {proc.returncode}\n"
                    + "\n".join(logs[name])
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                last = f"status={resp.status_code} body={resp.text[:300]}"
            except Exception as exc:  # noqa: BLE001
                last = repr(exc)
            time.sleep(3)
        raise TimeoutError(f"Timed out waiting for {url}: {last}")

    def try_json(url: str):
        try:
            return requests.get(url, timeout=5).json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable_after_shutdown", "error": repr(exc)}

    try:
        launch(
            "dataflow",
            [
                "python",
                "-u",
                "-m",
                "astraflow",
                "--config",
                str(experiment_yaml),
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        wait_json("http://127.0.0.1:8000/status", "dataflow", timeout_s=120)

        launch(
            "raas",
            [
                "python",
                "-u",
                "-m",
                "astraflow.raas.server",
                "--host",
                "127.0.0.1",
                "--port",
                "19190",
                "--config",
                str(experiment_yaml),
                "--config",
                str(raas_yaml),
                "--engine-id",
                "smoke-raas",
                "--astraflow-url",
                "http://127.0.0.1:8000",
            ],
            {"CUDA_VISIBLE_DEVICES": raas_visible_devices},
        )

        raas_status = wait_json("http://127.0.0.1:19190/status", "raas", timeout_s=25 * 60)

        deadline = time.time() + 5 * 60
        dataflow_status = {}
        while time.time() < deadline:
            dataflow_status = requests.get(
                "http://127.0.0.1:8000/status",
                timeout=5,
            ).json()
            if dataflow_status.get("raas_pool", {}).get("size", 0) >= 1:
                break
            time.sleep(3)
        else:
            raise TimeoutError("RaaS did not self-register with AstraFlow")

        from astraflow.dataflow.raas2_engine import dumps_object, loads_object

        ready = requests.post(
            "http://127.0.0.1:8000/ready",
            data=dumps_object({"train_batch_size": 1, "model_id": "model0"}),
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        ready.raise_for_status()

        batch_response = requests.get(
            "http://127.0.0.1:8000/batch",
            params={"model_id": "model0", "version": 0},
            timeout=20 * 60,
        )
        batch_response.raise_for_status()

        batch = loads_object(batch_response.content)
        tensor_shapes = {
            key: list(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        }

        return _json.dumps(
            {
                "mode": "dataflow_raas_batch_smoke",
                "dataflow_status": dataflow_status,
                "raas_status": raas_status,
                "batch_keys": sorted(batch.keys()),
                "tensor_shapes": tensor_shapes,
                "dataflow_log_tail": list(logs["dataflow"])[-80:],
                "raas_log_tail": list(logs["raas"])[-120:],
            },
            indent=2,
            sort_keys=True,
        )
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=30)


@app.function(
    image=astraflow_image,
    gpu=FULL_CHAIN_GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=2 * 60 * 60,
)
def dataflow_raas_trainer_smoke(
    model_path: str = "Qwen/Qwen3-0.6B",
    total_train_steps: int = 1,
    enable_vortex: bool = True,
    raas_data_parallel_size: int = 1,
    max_new_tokens: int = 8,
    dataset_prompt: str = "What is 1+1? Put the final answer in \\boxed{}.",
    dataset_answer: str = "\\boxed{2}",
) -> str:
    import json as _json
    import os
    import signal
    import subprocess
    import sys
    import textwrap
    import threading
    import time
    from collections import deque
    from pathlib import Path

    import requests

    root = Path("/tmp/astraflow_full_chain_smoke")
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    trainer_gpu_id = str(raas_data_parallel_size)
    raas_visible_devices = ",".join(str(i) for i in range(raas_data_parallel_size))
    prompt_literal = repr(dataset_prompt)
    answer_literal = repr(dataset_answer)

    dataset_py = root / "smoke_dataset.py"
    dataset_source = textwrap.dedent(
        """
            from datasets import Dataset


            def get_dataset(tokenizer=None, max_length=None, max_samples=None, **kwargs):
                n = max_samples or 16
                rows = {
                    "messages": [
                        [{"role": "user", "content": __PROMPT_LITERAL__}]
                        for _ in range(n)
                    ],
                    "answer": [__ANSWER_LITERAL__] * n,
                    "source": ["smoke"] * n,
                    "query_id": ["smoke-" + str(i) for i in range(n)],
                }
                return Dataset.from_dict(rows)
            """
    )
    dataset_source = dataset_source.replace("__ANSWER_LITERAL__", answer_literal)
    dataset_source = dataset_source.replace("__PROMPT_LITERAL__", prompt_literal)
    dataset_py.write_text(dataset_source, encoding="utf-8")

    experiment_yaml = root / "experiment.yaml"
    variant = "vortex" if enable_vortex else "baseline"
    experiment_yaml.write_text(
        textwrap.dedent(
            f"""
            experiment:
              experiment_name: astraflow-smoke
              trial_name: qwen3-0.6b-{variant}-{total_train_steps}step-full-chain
              fileroot: "{root}/run"
              model_path: "{model_path}"
              tokenizer_path: "{model_path}"
              seed: 1
              dtype: bfloat16
              weight_transfer_mode: tcp
              weight_transfer_strategies: full

            raas:
              models:
                model0:
                  backend: sglang
                  gconfig:
                    n_samples: 1
                    temperature: 0.0
                    max_new_tokens: {max_new_tokens}
                    min_new_tokens: 0
              delta_full_sync_interval: 0

            dataflow:
              host: "127.0.0.1"
              port: 8000
              buffer:
                size: 64
                replay_size: 0
                replay_ratio: 0
                max_staleness: 8
                filter_function: keep_all
              rollout_dataset:
                dataset_fn: "smoke_dataset:get_dataset"
                max_length: 128
                max_samples: 16
                batch_size: 1
              workflow_spec:
                workflow_cls: "rlvr"
                reward_fn: "math_verify"
                enable_thinking: false

            trainer_base:
              total_train_steps: {total_train_steps}
              train_batch_size: 1
              n_samples: 1
              temperature: 0.0
              engine:
                backend: fsdp
                data_parallel_size: 1
              actor:
                attn_impl: sdpa
                gradient_checkpointing: false
                mb_spec:
                  max_tokens_per_mb: 512
                optimizer:
                  type: adam
                  lr: 1e-6
                  weight_decay: 0.0
                  beta1: 0.9
                  beta2: 0.999
                  eps: 1e-8
                  lr_scheduler_type: constant
                  gradient_clipping: 1.0
                m2_threshold: 0.01
                eps_clip: 100.0
                eps_clip_higher: 100.0
                reward_scaling: 1
                reward_bias: 0
                kl_ctl: 0.0
                kl_penalty_coef: 0.0
                ppo_n_minibatches: 1
                reward_norm:
                  mean_level: batch
                  std_level: null
                adv_norm:
                  mean_level: batch
                  std_level: null
              ref:
                attn_impl: sdpa
                mb_spec:
                  max_tokens_per_mb: 512
              recover:
                mode: disabled
                freq_steps: 1000
              evaluator:
                eval_at_start: false
                freq_steps: 1000
              stats_logger:
                wandb:
                  mode: disabled

            trainer_model0:
              model_id: model0
            """
        ),
        encoding="utf-8",
    )

    raas_yaml = root / "raas.yaml"
    vortex_yaml = ""
    if enable_vortex:
        vortex_yaml = textwrap.indent(
            textwrap.dedent(
                f"""
                vortex:
                  module_name: gqa_block_sparse_attention
                  topk_val: 30
                  block_size: 16
                  attention_backend: flashinfer
                  impl_backend: triton
                  workload_chunk_size: 32
                  dtype: bfloat16
                  layers_skip:
                    - 0
                  block_reserved_bos: 1
                  block_reserved_eos: 1
                  max_seq_lens: 512
                  max_topk_val: 64
                  compilation_cache_dir: "{root}/vortex_cache"
                """
            ).strip(),
            "              ",
        )
    raas_yaml.write_text(
        textwrap.dedent(
            f"""
            rollout:
              max_concurrent_rollouts: 4
              max_concurrent_evals: 1
              pause_grace_period: 1
              enable_adaptive_availability: true
              target_waiting_queue_per_dp: 2
              adaptive_step_size: 1
              load_cache_ttl_ms: 100

            engine:
              model0:
                backend: sglang
                data_parallel_size: {raas_data_parallel_size}
                tensor_parallel_size: 1

            sglang:
              context_length: 512
              page_size: 16
              mem_fraction_static: 0.65
              max_running_requests: 4
              skip_tokenizer_init: false
              disable_cuda_graph: true
              disable_overlap_schedule: true
              attention_backend: flashinfer
{vortex_yaml}
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:/workspace/astraflow:" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    env["ASTRAFLOW_URL"] = "http://127.0.0.1:8000"
    env["ASTRAFLOW_RAAS_URL"] = "http://127.0.0.1:19190"
    env["WEIGHT_TRANSFER_HTTP_PORT"] = "19861"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29541"

    logs: dict[str, deque[str]] = {
        "dataflow": deque(maxlen=1200),
        "raas": deque(maxlen=2400),
        "trainer": deque(maxlen=1200),
    }
    procs: dict[str, subprocess.Popen] = {}

    def launch(name: str, cmd: list[str], extra_env: dict[str, str] | None = None):
        proc_env = env.copy()
        if extra_env:
            proc_env.update(extra_env)
        proc = subprocess.Popen(
            cmd,
            cwd="/workspace/astraflow",
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        procs[name] = proc

        def drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logs[name].append(line.rstrip())

        threading.Thread(target=drain, daemon=True).start()
        return proc

    def wait_json(url: str, name: str, timeout_s: float = 900.0):
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            proc = procs.get(name)
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited early with code {proc.returncode}\n"
                    + "\n".join(logs[name])
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                last = f"status={resp.status_code} body={resp.text[:300]}"
            except Exception as exc:  # noqa: BLE001
                last = repr(exc)
            time.sleep(3)
        raise TimeoutError(f"Timed out waiting for {url}: {last}")

    def try_json(url: str):
        try:
            return requests.get(url, timeout=5).json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable_after_shutdown", "error": repr(exc)}

    try:
        launch(
            "dataflow",
            [
                "python",
                "-u",
                "-m",
                "astraflow",
                "--config",
                str(experiment_yaml),
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        wait_json("http://127.0.0.1:8000/status", "dataflow", timeout_s=120)

        launch(
            "raas",
            [
                "python",
                "-u",
                "-m",
                "astraflow.raas.server",
                "--host",
                "127.0.0.1",
                "--port",
                "19190",
                "--config",
                str(experiment_yaml),
                "--config",
                str(raas_yaml),
                "--engine-id",
                "smoke-raas",
                "--astraflow-url",
                "http://127.0.0.1:8000",
            ],
            {"CUDA_VISIBLE_DEVICES": raas_visible_devices},
        )
        raas_status = wait_json("http://127.0.0.1:19190/status", "raas", timeout_s=25 * 60)

        deadline = time.time() + 5 * 60
        dataflow_status = {}
        while time.time() < deadline:
            dataflow_status = requests.get(
                "http://127.0.0.1:8000/status",
                timeout=5,
            ).json()
            if dataflow_status.get("raas_pool", {}).get("size", 0) >= 1:
                break
            time.sleep(3)
        else:
            raise TimeoutError("RaaS did not self-register with AstraFlow")

        trainer = launch(
            "trainer",
            [
                "torchrun",
                "--nnodes",
                "1",
                "--nproc-per-node",
                "1",
                "--master-addr",
                "127.0.0.1",
                "--master-port",
                "29541",
                "examples/launch_trainer.py",
                "--config",
                str(experiment_yaml),
                "--trainer",
                "trainer_model0",
            ],
            {"CUDA_VISIBLE_DEVICES": trainer_gpu_id},
        )
        try:
            trainer.wait(timeout=50 * 60)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Trainer did not finish one step within timeout") from exc
        if trainer.returncode != 0:
            raise RuntimeError(
                f"trainer exited with code {trainer.returncode}\n"
                + "\n".join(logs["trainer"])
            )

        final_dataflow_status = try_json("http://127.0.0.1:8000/status")
        final_raas_status = try_json("http://127.0.0.1:19190/status")
        evidence_terms = (
            "Launching SGLang server command",
            "vortex",
            "Auto-applying patches",
            "notify_version",
        )
        raas_evidence = [
            line
            for line in logs["raas"]
            if any(term.lower() in line.lower() for term in evidence_terms)
        ]

        return _json.dumps(
            {
                "mode": "dataflow_raas_trainer_smoke",
                "enable_vortex": enable_vortex,
                "total_train_steps": total_train_steps,
                "raas_data_parallel_size": raas_data_parallel_size,
                "max_new_tokens": max_new_tokens,
                "dataset_prompt": dataset_prompt,
                "dataset_answer": dataset_answer,
                "initial_dataflow_status": dataflow_status,
                "initial_raas_status": raas_status,
                "final_dataflow_status": final_dataflow_status,
                "final_raas_status": final_raas_status,
                "raas_evidence": raas_evidence[-120:],
                "dataflow_log_tail": list(logs["dataflow"])[-100:],
                "raas_log_tail": list(logs["raas"])[-140:],
                "trainer_log_tail": list(logs["trainer"])[-180:],
            },
            indent=2,
            sort_keys=True,
        )
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=30)
