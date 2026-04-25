from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import pandas as pd

from ml.models import ValidationMetrics, serialize_model_params
from ml.runtime import (
    apply_runtime_model_params,
    clamp_parallel_jobs,
    get_runtime_inventory,
    model_uses_gpu,
    resolve_cpu_threads_per_job,
)
from ml.validation import WalkForwardConfig, evaluate_holdout_validation, evaluate_walk_forward_validation


@dataclass(slots=True)
class TuningConfig:
    trials: int = 20
    metric: str = "spearman_ic"
    direction: str = "maximize"
    timeout_seconds: int | None = None
    seed: int = 42
    keep_top_trials: int = 5
    parallel_jobs: int | None = None
    gpu_devices: list[str] | None = None
    cpu_threads_per_trial: int | None = None


@dataclass(slots=True)
class TuningResult:
    enabled: bool
    metric: str
    direction: str
    best_score: float | None
    best_params: dict[str, object]
    trials: int
    completed_trials: int
    trial_records: list[dict[str, object]]
    parallel_jobs: int = 1
    cpu_threads_per_trial: int = 1
    gpu_devices: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "direction": self.direction,
            "best_score": self.best_score,
            "best_params": serialize_model_params(self.best_params),
            "trials": self.trials,
            "completed_trials": self.completed_trials,
            "trial_records": list(self.trial_records),
            "parallel_jobs": self.parallel_jobs,
            "cpu_threads_per_trial": self.cpu_threads_per_trial,
            "gpu_devices": list(self.gpu_devices or []),
        }


def tune_model_params(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    base_model_params: dict | None,
    tuning_config: TuningConfig,
    validation_mode: str,
    train_end_date: str | None,
    valid_start_date: str | None,
    valid_end_date: str | None,
    walk_forward_config: WalkForwardConfig | None,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> TuningResult:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Optuna is required for tuning. Install it with `py -m pip install optuna`.") from exc

    if tuning_config.trials < 1:
        raise ValueError("Optuna trial count must be at least 1.")
    fixed_params = dict(base_model_params or {})
    runtime = get_runtime_inventory(preferred_gpu_devices=tuning_config.gpu_devices or None)
    uses_gpu = model_uses_gpu(model_name, fixed_params)
    parallel_jobs = _resolve_tuning_parallel_jobs(
        tuning_config=tuning_config,
        uses_gpu=uses_gpu,
        available_gpu_devices=list(runtime.gpu_devices),
        logical_cpu_count=runtime.logical_cpu_count,
    )
    cpu_threads_per_trial = (
        max(1, int(tuning_config.cpu_threads_per_trial))
        if tuning_config.cpu_threads_per_trial is not None
        else resolve_cpu_threads_per_job(
            logical_cpu_count=runtime.logical_cpu_count,
            concurrent_jobs=parallel_jobs,
        )
    )
    sampler = optuna.samplers.TPESampler(seed=int(tuning_config.seed))
    study = optuna.create_study(direction=tuning_config.direction, sampler=sampler)
    trial_records: list[dict[str, object]] = []
    trial_lock = Lock()

    def objective(trial) -> float:
        params = _suggest_model_params(model_name=model_name, trial=trial, fixed_params=fixed_params)
        assigned_gpu_device = _assigned_gpu_device(
            uses_gpu=uses_gpu,
            available_gpu_devices=list(runtime.gpu_devices),
            trial_number=int(trial.number),
        )
        params = apply_runtime_model_params(
            model_name=model_name,
            model_params=params,
            cpu_threads=cpu_threads_per_trial,
            gpu_device=assigned_gpu_device,
        )
        if validation_mode == "walk_forward":
            if walk_forward_config is None:
                raise ValueError("walk_forward_config is required when validation_mode=walk_forward.")
            summary = evaluate_walk_forward_validation(
                frame=frame,
                feature_columns=feature_columns,
                label_column=label_column,
                model_name=model_name,
                model_params=params,
                config=walk_forward_config,
            )
        else:
            summary = evaluate_holdout_validation(
                frame=frame,
                feature_columns=feature_columns,
                label_column=label_column,
                model_name=model_name,
                model_params=params,
                train_end_date=train_end_date,
                valid_start_date=valid_start_date,
                valid_end_date=valid_end_date,
                purge_size=purge_size,
                embargo_size=embargo_size,
            )
        score = _metric_value(summary.metrics, tuning_config.metric)
        with trial_lock:
            trial_records.append(
                {
                    "trial_number": int(trial.number),
                    "score": float(score),
                    "params": serialize_model_params(params),
                    "fold_scores": [
                        float(fold.get(tuning_config.metric, 0.0))
                        for fold in summary.folds
                        if tuning_config.metric in fold
                    ],
                    "fold_count": int(summary.fold_count),
                }
            )
        return score

    study.optimize(
        objective,
        n_trials=int(tuning_config.trials),
        timeout=tuning_config.timeout_seconds,
        n_jobs=parallel_jobs,
    )
    return TuningResult(
        enabled=True,
        metric=tuning_config.metric,
        direction=tuning_config.direction,
        best_score=float(study.best_value) if study.trials else None,
        best_params={**fixed_params, **dict(study.best_params)},
        trials=int(tuning_config.trials),
        completed_trials=len(study.trials),
        trial_records=_select_top_trial_records(
            trial_records=trial_records,
            direction=tuning_config.direction,
            keep_top_trials=tuning_config.keep_top_trials,
        ),
        parallel_jobs=parallel_jobs,
        cpu_threads_per_trial=cpu_threads_per_trial,
        gpu_devices=list(runtime.gpu_devices),
    )


def _metric_value(metrics: ValidationMetrics, metric: str) -> float:
    mapping = metrics.as_dict()
    if metric not in mapping:
        available = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported tuning metric: {metric}. Available metrics: {available}")
    return float(mapping[metric])


def _suggest_model_params(model_name: str, trial, fixed_params: dict[str, object]) -> dict[str, object]:
    if model_name == "ridge":
        params = dict(fixed_params)
        if "alpha" not in params:
            params["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        return params
    if model_name == "xgboost":
        params = dict(fixed_params)
        params.setdefault("device", "cpu")
        params.setdefault("tree_method", "hist")
        if "n_estimators" not in params:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=50)
        if "max_depth" not in params:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        if "learning_rate" not in params:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        if "subsample" not in params:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        if "colsample_bytree" not in params:
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        if "reg_lambda" not in params:
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return params
    if model_name == "catboost":
        params = dict(fixed_params)
        params.setdefault("task_type", "CPU")
        params.setdefault("loss_function", "RMSE")
        if "iterations" not in params:
            params["iterations"] = trial.suggest_int("iterations", 100, 600, step=50)
        if "depth" not in params:
            params["depth"] = trial.suggest_int("depth", 3, 8)
        if "learning_rate" not in params:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        if "l2_leaf_reg" not in params:
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True)
        return params
    raise ValueError(f"Unsupported ML model for tuning: {model_name}")


def _select_top_trial_records(
    *,
    trial_records: list[dict[str, object]],
    direction: str,
    keep_top_trials: int,
) -> list[dict[str, object]]:
    if keep_top_trials < 1:
        raise ValueError("keep_top_trials must be >= 1.")
    reverse = direction != "minimize"
    ranked = sorted(trial_records, key=lambda item: float(item["score"]), reverse=reverse)
    return ranked[:keep_top_trials]


def _resolve_tuning_parallel_jobs(
    *,
    tuning_config: TuningConfig,
    uses_gpu: bool,
    available_gpu_devices: list[str],
    logical_cpu_count: int,
) -> int:
    if uses_gpu:
        gpu_capacity = max(1, len(available_gpu_devices) or 1)
        return clamp_parallel_jobs(tuning_config.parallel_jobs, max_jobs=gpu_capacity)
    cpu_capacity = max(1, logical_cpu_count)
    return clamp_parallel_jobs(tuning_config.parallel_jobs, max_jobs=cpu_capacity)


def _assigned_gpu_device(
    *,
    uses_gpu: bool,
    available_gpu_devices: list[str],
    trial_number: int,
) -> str | None:
    if not uses_gpu or not available_gpu_devices:
        return None
    return available_gpu_devices[trial_number % len(available_gpu_devices)]
