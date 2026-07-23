#!/usr/bin/env python3
"""Run a paired dense/Vortex SGLang serving benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests


BENCHMARK_SUMMARY_FIELDS = (
    "backend",
    "dataset_name",
    "completed",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "total_output_tokens_retokenized",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "concurrency",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the same SGLang model in dense and Vortex modes, then run "
            "SGLang's official serving benchmark with identical random requests."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--modes", default="dense,sparse")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max-topk", type=int, default=256)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--quality-prompt",
        default=(
            "Compute 17 * 23. Give one short sentence of reasoning, then the "
            "final integer."
        ),
    )
    parser.add_argument("--quality-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Additional single SGLang CLI argument; repeat as needed.",
    )
    return parser.parse_args()


def _visible_gpu_index(explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," not in visible and visible.isdigit():
        return int(visible)
    return None


def _server_environment(gpu_index: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _gpu_memory_mib(gpu_index: int | None) -> int | None:
    if gpu_index is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


class MemoryMonitor:
    def __init__(self, gpu_index: int | None, interval: float = 0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.gpu_index is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _gpu_memory_mib(self.gpu_index)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak_mib(self) -> int | None:
        return max(self.samples) if self.samples else None


def _vortex_config(args: argparse.Namespace) -> str:
    config = {
        "module_name": "gqa_block_sparse_attention",
        "topk_val": args.topk,
        "max_topk_val": args.max_topk,
        "block_size": args.page_size,
        "attention_backend": "flashinfer",
        "impl_backend": "triton",
        "dtype": "bfloat16",
        "layers_skip": [0],
        "block_reserved_bos": 1,
        "block_reserved_eos": 1,
        "max_seq_lens": args.context_length,
    }
    return json.dumps(config, separators=(",", ":"))


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        args.python,
        "-m",
        "astraflow.raas.entrypoint",
        "--model-path",
        args.model,
        "--tokenizer-path",
        args.tokenizer or args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--attention-backend",
        "flashinfer",
        "--page-size",
        str(args.page_size),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(max(args.max_concurrency, 4)),
        "--dtype",
        "bfloat16",
        "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]
    if mode == "sparse":
        command.extend(["--vortex-config", _vortex_config(args)])
    command.extend(args.server_extra_arg)
    return command


def _wait_until_healthy(
    process: subprocess.Popen[str],
    base_url: str,
    timeout: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = _tail(log_path)
            raise RuntimeError(
                f"SGLang exited before becoming healthy (rc={process.returncode}).\n{tail}"
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"SGLang health timeout: {last_error}\n{_tail(log_path)}")


def _quality_probe(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": args.quality_prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.quality_max_new_tokens,
            },
            "stream": False,
        },
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    body = response.json()
    text = body.get("text", "") if isinstance(body, dict) else str(body)
    return {
        "prompt": args.quality_prompt,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "meta_info": body.get("meta_info") if isinstance(body, dict) else None,
    }


def _run_official_benchmark(
    args: argparse.Namespace,
    mode: str,
    base_url: str,
    result_path: Path,
) -> tuple[dict[str, Any], str]:
    result_path.unlink(missing_ok=True)
    command = [
        args.python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random-ids",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        "inf",
        "--warmup-requests",
        str(args.warmup_requests),
        "--seed",
        str(args.seed),
        "--tokenize-prompt",
        "--disable-tqdm",
        "--output-details",
        "--output-file",
        str(result_path),
        "--tag",
        mode,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sglang.bench_serving failed for {mode} (rc={completed.returncode})\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    lines = [line for line in result_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Benchmark produced no JSON result: {result_path}")
    return json.loads(lines[-1]), completed.stdout


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _summarize_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: benchmark[key]
        for key in BENCHMARK_SUMMARY_FIELDS
        if key in benchmark
    }
    errors = benchmark.get("errors")
    if isinstance(errors, list):
        summary["error_count"] = sum(bool(error) for error in errors)
    return summary


def _tail(path: Path, line_count: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    mode_dir = args.output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = mode_dir / "server.log"
    benchmark_path = mode_dir / "benchmark.jsonl"
    stdout_path = mode_dir / "benchmark.stdout.txt"
    base_url = f"http://{args.host}:{args.port}"
    gpu_index = _visible_gpu_index(args.gpu_index)
    command = _server_command(args, mode)
    server_env = _server_environment(gpu_index)
    memory_before_mib = _gpu_memory_mib(gpu_index)

    with server_log_path.open("w") as server_log:
        process = subprocess.Popen(
            command,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_until_healthy(
                process,
                base_url,
                args.startup_timeout,
                server_log_path,
            )
            memory_idle_mib = _gpu_memory_mib(gpu_index)
            quality = _quality_probe(args, base_url)
            monitor = MemoryMonitor(gpu_index)
            monitor.start()
            try:
                benchmark, benchmark_stdout = _run_official_benchmark(
                    args,
                    mode,
                    base_url,
                    benchmark_path,
                )
            finally:
                monitor.stop()
            stdout_path.write_text(benchmark_stdout)
        finally:
            _stop_process(process)

    return {
        "mode": mode,
        "server_command": command,
        "benchmark": _summarize_benchmark(benchmark),
        "quality": quality,
        "gpu_memory": {
            "physical_gpu_index": gpu_index,
            "before_server_mib": memory_before_mib,
            "idle_server_mib": memory_idle_mib,
            "benchmark_peak_mib": monitor.peak_mib,
        },
        "artifacts": {
            "server_log": str(server_log_path),
            "benchmark_jsonl": str(benchmark_path),
            "benchmark_stdout": str(stdout_path),
        },
    }


def _comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense = results.get("dense")
    sparse = results.get("sparse")
    if dense is None or sparse is None:
        return {}

    dense_bench = dense["benchmark"]
    sparse_bench = sparse["benchmark"]

    def ratio(key: str) -> float | None:
        dense_value = dense_bench.get(key)
        sparse_value = sparse_bench.get(key)
        if not dense_value or sparse_value is None:
            return None
        return float(sparse_value) / float(dense_value)

    return {
        "quality_text_exact_match": (
            dense["quality"]["text"] == sparse["quality"]["text"]
        ),
        "sparse_over_dense": {
            "request_throughput": ratio("request_throughput"),
            "output_throughput": ratio("output_throughput"),
            "mean_e2e_latency": ratio("mean_e2e_latency_ms"),
            "mean_ttft": ratio("mean_ttft_ms"),
            "mean_tpot": ratio("mean_tpot_ms"),
        },
    }


def main() -> int:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = set(modes) - {"dense", "sparse"}
    if invalid:
        raise ValueError(f"Unsupported modes: {sorted(invalid)}")
    if args.input_len + args.output_len > args.context_length:
        raise ValueError("input_len + output_len must not exceed context_length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        print(f"=== Running {mode} benchmark ===", flush=True)
        results[mode] = run_mode(args, mode)

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"quality_prompt"}
        },
        "results": results,
        "comparison": _comparison(results),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
