from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count, environ
import subprocess
from typing import Sequence


@dataclass(slots=True, frozen=True)
class RuntimeInventory:
    logical_cpu_count: int
    gpu_devices: tuple[str, ...]


def get_runtime_inventory(*, preferred_gpu_devices: Sequence[str] | None = None) -> RuntimeInventory:
    return RuntimeInventory(
        logical_cpu_count=max(1, cpu_count() or 1),
        gpu_devices=tuple(detect_gpu_devices(preferred_gpu_devices=preferred_gpu_devices)),
    )


def detect_gpu_devices(*, preferred_gpu_devices: Sequence[str] | None = None) -> list[str]:
    if preferred_gpu_devices is not None:
        return _normalize_gpu_devices(preferred_gpu_devices)

    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        if visible.strip() == "-1":
            return []
        return _normalize_gpu_devices(visible.split(","))

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=3,
        )
    except Exception:  # noqa: BLE001
        return []
    return _normalize_gpu_devices(output.splitlines())


def model_uses_gpu(model_name: str, model_params: dict[str, object] | None) -> bool:
    params = dict(model_params or {})
    if model_name == "xgboost":
        return "cuda" in str(params.get("device", "cpu")).lower()
    if model_name == "catboost":
        return str(params.get("task_type", "CPU")).upper() == "GPU"
    return False


def apply_runtime_model_params(
    *,
    model_name: str,
    model_params: dict[str, object] | None,
    cpu_threads: int | None = None,
    gpu_device: str | None = None,
) -> dict[str, object]:
    params = dict(model_params or {})
    if cpu_threads is not None:
        normalized_threads = max(1, int(cpu_threads))
        if model_name == "xgboost":
            params["n_jobs"] = normalized_threads
        elif model_name == "catboost":
            params["thread_count"] = normalized_threads

    if gpu_device is not None and model_uses_gpu(model_name, params):
        if model_name == "xgboost":
            params["device"] = f"cuda:{gpu_device}"
        elif model_name == "catboost":
            params["task_type"] = "GPU"
            params["devices"] = str(gpu_device)
    return params


def resolve_cpu_threads_per_job(*, logical_cpu_count: int, concurrent_jobs: int) -> int:
    return max(1, int(logical_cpu_count) // max(1, int(concurrent_jobs)))


def clamp_parallel_jobs(
    requested_jobs: int | None,
    *,
    max_jobs: int,
) -> int:
    if requested_jobs is None:
        return max(1, max_jobs)
    return max(1, min(int(requested_jobs), max(1, max_jobs)))


def _normalize_gpu_devices(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if token not in normalized:
            normalized.append(token)
    return normalized
