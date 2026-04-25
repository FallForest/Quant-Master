from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import shutil

import pandas as pd

from data.loading import load_bars_for_symbols
from data.provider_factory import build_data_provider
from ml.artifacts import load_signal_artifact, save_signal_artifact
from ml.dataset import build_training_dataset, drop_rows_without_features
from ml.diagnostics import build_overfit_diagnostics
from ml.experiments.loader import load_experiment_group_spec, load_experiment_spec
from ml.experiments.scheduler import GroupExecutionOptions, execute_group
from ml.experiments.specs import ExperimentGroupSpec, ExperimentSpec
from ml.models import evaluate_model
from ml.selection import score_candidate_metrics
from ml.tuning import TuningConfig
from ml.training import train_ml_signal_model
from ml.validation import WalkForwardConfig


def run_experiment_from_path(path: str | Path) -> dict[str, object]:
    spec = load_experiment_spec(path)
    return run_experiment(spec=spec, experiment_path=Path(path))


def run_experiment(spec: ExperimentSpec, *, experiment_path: Path | None = None) -> dict[str, object]:
    _validate_experiment_windows(spec)
    artifact_path = Path(spec.report.artifact_path or Path("artifacts") / "experiments" / spec.name / "artifact")
    report_dir = Path(spec.report.output_dir or Path("reports") / "experiments" / spec.name)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    initial_metadata = train_ml_signal_model(
        market=spec.market,
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        reference_root=spec.reference_root,
        adjust=spec.adjust,
        timeframe=spec.timeframe,
        symbols=spec.symbols or None,
        universe=spec.universe,
        start_date=spec.train.start_date,
        end_date=spec.train.end_date,
        artifact_path=str(artifact_path),
        model_name=spec.model,
        model_params=spec.model_params,
        feature_columns=spec.features,
        label_horizon=spec.train.label_horizon,
        target_mode=spec.train.target_mode,
        train_end_date=spec.train.train_end_date,
        valid_start_date=spec.train.valid_start_date,
        valid_end_date=spec.train.valid_end_date,
        validation_mode=spec.train.validation_mode,
        walk_forward_config=_build_walk_forward_config(spec),
        tuning_config=_build_tuning_config(spec),
        purge_size=spec.train.purge_size,
        embargo_size=spec.train.embargo_size,
    )

    training_metadata = initial_metadata
    if spec.train.tuning:
        candidate_selection = _run_candidate_selection(
            spec=spec,
            training_metadata=initial_metadata,
            artifact_root=report_dir / "_candidate_selection",
        )
        selected_params = dict(candidate_selection["selected_model_params"])
        selection_start = str(spec.train.valid_start_date)
        pre_selection_end = (pd.Timestamp(selection_start) - pd.Timedelta(days=1)).date().isoformat()
        training_metadata = train_ml_signal_model(
            market=spec.market,
            provider_name=spec.provider,
            data_root=spec.data_root,
            universe_root=spec.universe_root,
            reference_root=spec.reference_root,
            adjust=spec.adjust,
            timeframe=spec.timeframe,
            symbols=spec.symbols or None,
            universe=spec.universe,
            start_date=spec.train.start_date,
            end_date=spec.train.end_date,
            artifact_path=str(artifact_path),
            model_name=spec.model,
            model_params=selected_params,
            feature_columns=spec.features,
            label_horizon=spec.train.label_horizon,
            target_mode=spec.train.target_mode,
            train_end_date=pre_selection_end,
            valid_start_date=spec.train.valid_start_date,
            valid_end_date=spec.train.valid_end_date,
            validation_mode="holdout",
            walk_forward_config=None,
            tuning_config=None,
            purge_size=spec.train.purge_size,
            embargo_size=spec.train.embargo_size,
        )
        training_metadata["tuning"] = dict(candidate_selection["tuning"])
        training_metadata["candidate_selection"] = dict(candidate_selection["candidate_selection"])
        _rewrite_artifact_metadata(artifact_path=artifact_path, metadata=training_metadata)

    signal_test = _run_signal_test(spec=spec, artifact_path=artifact_path)
    final_diagnostics = build_overfit_diagnostics(
        trial_records=list(dict(training_metadata.get("tuning", {})).get("trial_records", [])),
        direction=str(dict(training_metadata.get("tuning", {})).get("direction", "maximize")),
        returns=pd.Series(dtype=float),
    )

    summary = {
        "name": spec.name,
        "group": spec.group,
        "experiment_path": str(experiment_path) if experiment_path else None,
        "artifact_path": str(artifact_path),
        "report_dir": str(report_dir),
        "market": spec.market,
        "provider": spec.provider,
        "timeframe": spec.timeframe,
        "adjust": spec.adjust,
        "symbols": list(spec.symbols),
        "universe": spec.universe,
        "reference_root": spec.reference_root,
        "features": list(spec.features),
        "model": spec.model,
        "model_params": dict(training_metadata.get("model_params", spec.model_params)),
        "train": asdict(spec.train),
        "signal_test": asdict(spec.signal_test),
        "training_metadata": training_metadata,
        "validation_metrics": dict(training_metadata.get("validation_metrics", {})),
        "candidate_selection": dict(training_metadata.get("candidate_selection", {})),
        "signal_test_metrics": dict(signal_test["metrics"]),
        "signal_test_rows": int(signal_test["rows"]),
        "research_diagnostics": final_diagnostics.as_dict(),
    }
    (report_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_experiment_group_from_path(
    path: str | Path,
    *,
    continue_on_error: bool = False,
    execution_options: GroupExecutionOptions | None = None,
) -> dict[str, object]:
    group_path = Path(path)
    group_spec = load_experiment_group_spec(group_path)
    base_dir = group_path.parent
    experiment_paths = [str((base_dir / item).resolve()) if not Path(item).is_absolute() else item for item in group_spec.experiments]
    return run_experiment_group(
        group_spec=group_spec,
        experiment_paths=experiment_paths,
        continue_on_error=continue_on_error,
        execution_options=execution_options,
    )


def run_experiment_group(
    *,
    group_spec: ExperimentGroupSpec,
    experiment_paths: list[str],
    continue_on_error: bool = False,
    execution_options: GroupExecutionOptions | None = None,
) -> dict[str, object]:
    group_output_dir = Path(group_spec.output_dir or Path("reports") / "experiments" / "groups" / group_spec.name)
    group_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = group_output_dir / "group_summary.json"

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    def _write_group_summary() -> dict[str, object]:
        group_summary = {
            "name": group_spec.name,
            "experiment_paths": [str(path) for path in experiment_paths],
            "experiment_count": len(experiment_paths),
            "completed_count": len(summaries),
            "failed_count": len(failures),
            "experiments": summaries,
            "failures": failures,
        }
        summary_path.write_text(
            json.dumps(group_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return group_summary

    _write_group_summary()
    scheduled_summaries, scheduled_failures = execute_group(
        experiment_paths=experiment_paths,
        continue_on_error=continue_on_error,
        options=execution_options or GroupExecutionOptions(),
    )
    summaries.extend(scheduled_summaries)
    failures.extend(scheduled_failures)
    _write_group_summary()
    return _write_group_summary()


def _build_walk_forward_config(spec: ExperimentSpec) -> WalkForwardConfig | None:
    if spec.train.walk_forward is None:
        return None
    return WalkForwardConfig(
        train_size=spec.train.walk_forward.train_size,
        valid_size=spec.train.walk_forward.valid_size,
        step_size=spec.train.walk_forward.step_size,
        expanding=spec.train.walk_forward.expanding,
        purge_size=spec.train.walk_forward.purge_size,
        embargo_size=spec.train.walk_forward.embargo_size,
    )


def _build_tuning_config(spec: ExperimentSpec) -> TuningConfig | None:
    if spec.train.tuning is None:
        return None
    return TuningConfig(
        trials=spec.train.tuning.trials,
        metric=spec.train.tuning.metric,
        direction=spec.train.tuning.direction,
        timeout_seconds=spec.train.tuning.timeout_seconds,
        seed=spec.train.tuning.seed,
        keep_top_trials=spec.train.tuning.keep_top_trials,
        parallel_jobs=spec.train.tuning.parallel_jobs,
        gpu_devices=list(spec.train.tuning.gpu_devices),
    )


def _run_candidate_selection(
    *,
    spec: ExperimentSpec,
    training_metadata: dict[str, object],
    artifact_root: Path,
) -> dict[str, object]:
    tuning_summary = dict(training_metadata.get("tuning", {}))
    trial_records = list(tuning_summary.get("trial_records", []))
    if not trial_records:
        return {
            "selected_model_params": dict(training_metadata.get("model_params", {})),
            "candidate_selection": {"enabled": False, "candidates": [], "selected_metric": spec.train.candidate_selection.metric},
            "tuning": tuning_summary,
        }

    selection_start = str(spec.train.valid_start_date)
    selection_end = str(spec.train.valid_end_date)
    pre_selection_end = (pd.Timestamp(selection_start) - pd.Timedelta(days=1)).date().isoformat()

    candidate_rows: list[dict[str, object]] = []
    artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, record in enumerate(trial_records, start=1):
            params = dict(record.get("params", {}))
            candidate_artifact = artifact_root / f"candidate_{index}"
            train_ml_signal_model(
                market=spec.market,
                provider_name=spec.provider,
                data_root=spec.data_root,
                universe_root=spec.universe_root,
                reference_root=spec.reference_root,
                adjust=spec.adjust,
                timeframe=spec.timeframe,
                symbols=spec.symbols or None,
                universe=spec.universe,
                start_date=spec.train.start_date,
                end_date=pre_selection_end,
                artifact_path=str(candidate_artifact),
                model_name=spec.model,
                model_params=params,
                feature_columns=spec.features,
                label_horizon=spec.train.label_horizon,
                target_mode=spec.train.target_mode,
                validation_mode="holdout",
                tuning_config=None,
                purge_size=0,
                embargo_size=0,
            )
            selection_result = _evaluate_signal_window(
                spec=spec,
                artifact_path=candidate_artifact,
                start_date=selection_start,
                end_date=selection_end,
            )
            candidate_rows.append(
                {
                    "candidate_index": index,
                    "trial_number": record.get("trial_number"),
                    "tuning_score": record.get("score"),
                    "model_params": params,
                    "selection_signal_metrics": dict(selection_result["metrics"]),
                    "selection_rows": int(selection_result["rows"]),
                }
            )
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)

    selection_config = spec.train.candidate_selection
    ranked_rows = sorted(
        candidate_rows,
        key=lambda item: score_candidate_metrics(
            item["selection_signal_metrics"],
            metric=selection_config.metric,
            direction=selection_config.direction,
        ),
        reverse=True,
    )[: max(1, selection_config.top_k)]
    selected = ranked_rows[0]
    return {
        "selected_model_params": dict(selected["model_params"]),
        "candidate_selection": {
            "enabled": True,
            "config": selection_config.as_dict(),
            "candidates": ranked_rows,
            "selected_candidate_index": selected["candidate_index"],
            "selected_metric": selection_config.metric,
        },
        "tuning": tuning_summary,
    }


def _rewrite_artifact_metadata(*, artifact_path: Path, metadata: dict[str, object]) -> None:
    model, _ = load_signal_artifact(artifact_path)
    save_signal_artifact(artifact_path=artifact_path, model=model, metadata=metadata)


def _run_signal_test(*, spec: ExperimentSpec, artifact_path: Path) -> dict[str, object]:
    return _evaluate_signal_window(
        spec=spec,
        artifact_path=artifact_path,
        start_date=spec.signal_test.start_date,
        end_date=spec.signal_test.end_date,
    )


def _evaluate_signal_window(
    *,
    spec: ExperimentSpec,
    artifact_path: Path,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    estimator, metadata = load_signal_artifact(artifact_path)
    frame = _build_signal_frame(
        spec=spec,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
    )
    metrics = evaluate_model(
        estimator=estimator,
        frame=frame,
        feature_columns=list(metadata["feature_columns"]),
        label_column=str(metadata["label_column"]),
    )
    return {
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(frame)),
        "metrics": metrics.as_dict(),
    }


def _build_signal_frame(
    *,
    spec: ExperimentSpec,
    metadata: dict[str, object],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    provider = build_data_provider(
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        adjust=spec.adjust,
    )
    start = pd.Timestamp(start_date).to_pydatetime()
    end = pd.Timestamp(end_date).to_pydatetime()
    symbols = list(spec.symbols)
    if not symbols:
        symbols = provider.load_universe(market=spec.market, universe=spec.universe, date=start)
    data = load_bars_for_symbols(
        provider=provider,
        market=spec.market,
        timeframe=spec.timeframe,
        symbols=symbols,
        start=start,
        end=end,
        progress_desc="Evaluating signal window",
        show_progress=True,
        empty_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )
    bundle = build_training_dataset(
        data=data,
        label_horizon=int(metadata["label_horizon"]),
        feature_columns=list(metadata["feature_columns"]),
        reference_root=str(metadata.get("reference_root", spec.reference_root)),
        market=str(metadata.get("market", spec.market)),
        target_mode=str(metadata.get("target_mode", spec.train.target_mode)),
    )
    return drop_rows_without_features(
        frame=bundle.frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
    )


def _validate_experiment_windows(spec: ExperimentSpec) -> None:
    if spec.signal_test.start_date <= spec.train.end_date:
        raise ValueError("signal_test.start_date must be later than train.end_date so final testing stays fully out-of-sample.")
    if spec.train.tuning and spec.train.valid_end_date and spec.train.valid_end_date != spec.train.end_date:
        raise ValueError("When train.tuning is configured, train.valid_end_date must equal train.end_date.")
