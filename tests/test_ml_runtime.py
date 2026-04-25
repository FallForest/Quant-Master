from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.runtime import apply_runtime_model_params, resolve_cpu_threads_per_job


def test_apply_runtime_model_params_for_xgboost_gpu() -> None:
    params = apply_runtime_model_params(
        model_name="xgboost",
        model_params={"device": "cuda", "n_jobs": -1},
        cpu_threads=3,
        gpu_device="1",
    )
    assert params["device"] == "cuda:1"
    assert params["n_jobs"] == 3


def test_apply_runtime_model_params_for_catboost_gpu() -> None:
    params = apply_runtime_model_params(
        model_name="catboost",
        model_params={"task_type": "GPU", "thread_count": -1},
        cpu_threads=2,
        gpu_device="0",
    )
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"
    assert params["thread_count"] == 2


def test_resolve_cpu_threads_per_job() -> None:
    assert resolve_cpu_threads_per_job(logical_cpu_count=16, concurrent_jobs=4) == 4
