from __future__ import annotations

from pathlib import Path

import yaml

from ml.experiments.specs import (
    ExperimentGroupSpec,
    ExperimentReportSpec,
    ExperimentSignalTestSpec,
    ExperimentSpec,
    ExperimentTrainSpec,
    ExperimentTuningSpec,
    ExperimentWalkForwardSpec,
)
from ml.selection import CandidateSelectionConfig


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    train_raw = raw.get("train") or {}
    signal_test_raw = raw.get("signal_test") or raw.get("backtest") or {}
    report_raw = raw.get("report") or {}
    walk_forward_raw = train_raw.get("walk_forward") or None
    tuning_raw = train_raw.get("tuning") or None
    candidate_selection_raw = train_raw.get("candidate_selection") or {}
    return ExperimentSpec(
        name=str(raw["name"]),
        market=str(raw.get("market", "ashare")),
        provider=str(raw.get("provider", "parquet")),
        data_root=str(raw.get("data_root", "data/lake")),
        universe_root=str(raw.get("universe_root", "data/universe")),
        reference_root=str(raw.get("reference_root", "data/reference")),
        timeframe=str(raw.get("timeframe", "1d")),
        adjust=str(raw.get("adjust", "qfq")),
        symbols=[str(item) for item in raw.get("symbols", [])],
        universe=str(raw["universe"]) if raw.get("universe") is not None else None,
        features=[str(item) for item in raw.get("features", [])],
        model=str(raw.get("model", "ridge")),
        model_params=dict(raw.get("model_params") or {}),
        train=ExperimentTrainSpec(
            start_date=str(train_raw["start_date"]),
            end_date=str(train_raw["end_date"]),
            validation_mode=str(train_raw.get("validation_mode", "holdout")),
            train_end_date=str(train_raw["train_end_date"]) if train_raw.get("train_end_date") else None,
            valid_start_date=str(train_raw["valid_start_date"]) if train_raw.get("valid_start_date") else None,
            valid_end_date=str(train_raw["valid_end_date"]) if train_raw.get("valid_end_date") else None,
            label_horizon=int(train_raw.get("label_horizon", 5)),
            target_mode=str(train_raw.get("target_mode", "future_return")),
            purge_size=int(train_raw.get("purge_size", 0)),
            embargo_size=int(train_raw.get("embargo_size", 0)),
            walk_forward=ExperimentWalkForwardSpec(
                train_size=int(walk_forward_raw["train_size"]),
                valid_size=int(walk_forward_raw["valid_size"]),
                step_size=int(walk_forward_raw["step_size"]) if walk_forward_raw.get("step_size") is not None else None,
                expanding=bool(walk_forward_raw.get("expanding", True)),
                purge_size=int(walk_forward_raw.get("purge_size", train_raw.get("purge_size", 0))),
                embargo_size=int(walk_forward_raw.get("embargo_size", train_raw.get("embargo_size", 0))),
            )
            if walk_forward_raw
            else None,
            tuning=ExperimentTuningSpec(
                trials=int(tuning_raw.get("trials", 20)),
                metric=str(tuning_raw.get("metric", "spearman_ic")),
                direction=str(tuning_raw.get("direction", "maximize")),
                timeout_seconds=int(tuning_raw["timeout_seconds"]) if tuning_raw.get("timeout_seconds") is not None else None,
                seed=int(tuning_raw.get("seed", 42)),
                keep_top_trials=int(tuning_raw.get("keep_top_trials", 5)),
                parallel_jobs=int(tuning_raw["parallel_jobs"]) if tuning_raw.get("parallel_jobs") is not None else None,
                gpu_devices=[str(item) for item in tuning_raw.get("gpu_devices", [])],
            )
            if tuning_raw
            else None,
            candidate_selection=CandidateSelectionConfig(
                top_k=int(candidate_selection_raw.get("top_k", 5)),
                metric=str(candidate_selection_raw.get("metric", "oos_spearman_ic")),
                direction=str(candidate_selection_raw.get("direction", "maximize")),
            ),
        ),
        signal_test=ExperimentSignalTestSpec(
            start_date=str(signal_test_raw["start_date"]),
            end_date=str(signal_test_raw["end_date"]),
        ),
        report=ExperimentReportSpec(
            output_dir=str(report_raw["output_dir"]) if report_raw.get("output_dir") else None,
            artifact_path=str(report_raw["artifact_path"]) if report_raw.get("artifact_path") else None,
        ),
        group=str(raw["group"]) if raw.get("group") else None,
    )


def load_experiment_group_spec(path: str | Path) -> ExperimentGroupSpec:
    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return ExperimentGroupSpec(
        name=str(raw["name"]),
        experiments=[str(item) for item in raw.get("experiments", [])],
        output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
    )
