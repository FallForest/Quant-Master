from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.loading import load_bars_for_symbols
from data.provider_factory import build_data_provider
from ml.artifacts import load_signal_artifact, save_signal_artifact
from ml.dataset import build_training_dataset, drop_rows_without_features
from ml.experiments.compare import load_group_summary
from ml.experiments.group_tracking import GroupRunTracker
from ml.experiments.loader import load_experiment_group_spec, load_experiment_spec
from ml.experiments.specs import ExperimentSpec
from ml.models import aggregate_validation_metrics, evaluate_model
from ml.validation import ValidationSummary, build_walk_forward_splits, split_dataset_by_time


def backfill_signal_metrics_from_path(
    path: str | Path,
    *,
    artifact_path_override: str | Path | None = None,
    report_dir_override: str | Path | None = None,
) -> dict[str, object]:
    """按实验路径回填信号评估结果。"""

    experiment_path = Path(path)
    spec = load_experiment_spec(experiment_path)
    artifact_path = Path(
        artifact_path_override
        or spec.report.artifact_path
        or Path("artifacts") / "experiments" / spec.name / "artifact"
    )
    report_dir = Path(
        report_dir_override
        or spec.report.output_dir
        or Path("reports") / "experiments" / spec.name
    )
    return backfill_signal_metrics(
        spec=spec,
        artifact_path=artifact_path,
        report_dir=report_dir,
        experiment_path=experiment_path,
    )


def backfill_signal_metrics_group_from_path(path: str | Path, *, resume: bool = False) -> dict[str, object]:
    """批量回填一个实验组。"""

    group_path = Path(path)
    group_spec = load_experiment_group_spec(group_path)
    base_dir = group_path.parent
    experiment_paths = [
        (base_dir / item).resolve() if not Path(item).is_absolute() else Path(item)
        for item in group_spec.experiments
    ]
    group_output_dir = Path(group_spec.output_dir or Path("reports") / "experiments" / "groups" / group_spec.name)
    summary_path = group_output_dir / "group_summary.json"
    existing_summary = load_group_summary(group_path) if summary_path.exists() else None
    existing_index = {
        str(item.get("experiment_path")): dict(item)
        for item in list((existing_summary or {}).get("experiments", []))
    }
    tracker = GroupRunTracker(
        group_name=group_spec.name,
        experiment_paths=[str(path) for path in experiment_paths],
        output_dir=group_output_dir,
        mode="backfill",
        continue_on_error=False,
        resume=resume,
        execution_options={},
    )
    pending_paths = [Path(path) for path in tracker.pending_paths()]
    if not pending_paths:
        return tracker.finalize(status="completed")

    try:
        for experiment_path in pending_paths:
            summary = backfill_signal_metrics_from_path(
                experiment_path,
                artifact_path_override=existing_index.get(str(experiment_path), {}).get("artifact_path"),
                report_dir_override=existing_index.get(str(experiment_path), {}).get("report_dir"),
            )
            tracker.record_completed(summary)
    except Exception as exc:
        tracker.record_failure(experiment_path=str(experiment_path), error=str(exc))
        tracker.finalize(status="failed")
        raise
    return tracker.finalize(status="completed")


def backfill_signal_metrics(
    *,
    spec: ExperimentSpec,
    artifact_path: Path,
    report_dir: Path,
    experiment_path: Path | None = None,
) -> dict[str, object]:
    """对单个实验执行实际 backfill。"""

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact path was not found for backfill: {artifact_path}. "
            "Run the experiment first or provide a summary with the correct artifact_path."
        )
    # backfill 复用现有模型对象，只更新由数据/评估口径决定的统计字段。
    model, metadata = load_signal_artifact(artifact_path)
    frame = _build_training_frame(spec=spec, metadata=metadata)
    validation_summary = _evaluate_existing_artifact_validation(
        estimator=model,
        metadata=metadata,
        frame=frame,
    )

    updated_metadata = dict(metadata)
    updated_metadata["validation_fold_count"] = validation_summary.fold_count
    updated_metadata["validation_rows"] = validation_summary.valid_rows
    updated_metadata["validation_metrics"] = validation_summary.metrics.as_dict()
    updated_metadata["validation_folds"] = validation_summary.folds
    save_signal_artifact(artifact_path=artifact_path, model=model, metadata=updated_metadata)

    summary_path = report_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "name": spec.name,
            "group": spec.group,
            "experiment_path": str(experiment_path) if experiment_path else None,
            "artifact_path": str(artifact_path),
            "report_dir": str(report_dir),
        }
    training_metadata = dict(summary.get("training_metadata", {}))
    training_metadata.update(updated_metadata)
    summary["experiment_path"] = str(experiment_path) if experiment_path else summary.get("experiment_path")
    summary["artifact_path"] = str(artifact_path)
    summary["report_dir"] = str(report_dir)
    summary["training_metadata"] = training_metadata
    summary["validation_metrics"] = dict(updated_metadata["validation_metrics"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _build_training_frame(*, spec: ExperimentSpec, metadata: dict[str, object]) -> pd.DataFrame:
    """按 artifact metadata 重新构造训练数据表。"""

    provider = build_data_provider(
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        adjust=spec.adjust,
    )
    start = pd.Timestamp(str(metadata["data_start"])).to_pydatetime()
    end = pd.Timestamp(str(metadata["data_end"])).to_pydatetime()
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
        progress_desc="Backfilling signal metrics",
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
        feature_normalization=str(metadata.get("feature_normalization", spec.feature_normalization)),
    )
    return drop_rows_without_features(
        frame=bundle.frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
    )


def _evaluate_existing_artifact_validation(
    *,
    estimator,
    metadata: dict[str, object],
    frame: pd.DataFrame,
) -> ValidationSummary:
    """根据 artifact 中记录的验证方式，重算 validation summary。"""

    feature_columns = list(metadata["feature_columns"])
    label_column = str(metadata["label_column"])
    validation_mode = str(metadata.get("validation_mode", "holdout"))
    if validation_mode == "walk_forward":
        walk_forward_config = metadata.get("walk_forward_config") or {}
        return _evaluate_existing_walk_forward(
            estimator=estimator,
            frame=frame,
            feature_columns=feature_columns,
            label_column=label_column,
            config=walk_forward_config,
        )
    return _evaluate_existing_holdout(
        estimator=estimator,
        frame=frame,
        feature_columns=feature_columns,
        label_column=label_column,
        metadata=metadata,
    )


def _evaluate_existing_holdout(
    *,
    estimator,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    metadata: dict[str, object],
) -> ValidationSummary:
    """用已有模型重算 holdout 验证指标。"""

    train_end_date, valid_start_date, valid_end_date = _resolve_holdout_dates(metadata)
    train_frame, valid_frame = split_dataset_by_time(
        frame=frame,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
        purge_size=int(metadata.get("purge_size", 0)),
        embargo_size=int(metadata.get("embargo_size", 0)),
    )
    metrics = evaluate_model(
        estimator=estimator,
        frame=valid_frame,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    folds = [
        {
            "fold_index": 1,
            "train_start": _frame_date(train_frame, first=True),
            "train_end": _frame_date(train_frame, first=False),
            "valid_start": _frame_date(valid_frame, first=True),
            "valid_end": _frame_date(valid_frame, first=False),
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            **metrics.as_dict(),
        }
    ]
    return ValidationSummary(
        mode="holdout",
        metrics=metrics,
        folds=folds,
        fold_count=1,
        train_rows=int(len(train_frame)),
        valid_rows=int(len(valid_frame)),
    )


def _evaluate_existing_walk_forward(
    *,
    estimator,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    config: dict[str, object],
) -> ValidationSummary:
    """用已有模型重算 walk-forward 验证指标。"""

    splits = build_walk_forward_splits(
        frame,
        config=_build_walk_forward_config(config),
    )
    fold_metrics = []
    folds: list[dict[str, object]] = []
    total_train_rows = 0
    total_valid_rows = 0
    for index, (train_frame, valid_frame) in enumerate(splits, start=1):
        metrics = evaluate_model(
            estimator=estimator,
            frame=valid_frame,
            feature_columns=feature_columns,
            label_column=label_column,
        )
        fold_metrics.append(metrics)
        total_train_rows += len(train_frame)
        total_valid_rows += len(valid_frame)
        folds.append(
            {
                "fold_index": index,
                "train_start": _frame_date(train_frame, first=True),
                "train_end": _frame_date(train_frame, first=False),
                "valid_start": _frame_date(valid_frame, first=True),
                "valid_end": _frame_date(valid_frame, first=False),
                "train_rows": int(len(train_frame)),
                "valid_rows": int(len(valid_frame)),
                **metrics.as_dict(),
            }
        )
    metrics = aggregate_validation_metrics(fold_metrics)
    return ValidationSummary(
        mode="walk_forward",
        metrics=metrics,
        folds=folds,
        fold_count=len(folds),
        train_rows=int(total_train_rows),
        valid_rows=int(total_valid_rows),
    )


def _resolve_holdout_dates(metadata: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    """从 artifact metadata 里反推出 holdout 分割日期。"""

    folds = list(metadata.get("validation_folds", []))
    if folds:
        fold = dict(folds[0])
        return (
            str(fold.get("train_end")) if fold.get("train_end") else None,
            str(fold.get("valid_start")) if fold.get("valid_start") else None,
            str(fold.get("valid_end")) if fold.get("valid_end") else None,
        )
    tuning = dict(metadata.get("tuning", {}))
    return (
        None,
        str(tuning.get("selection_valid_start")) if tuning.get("selection_valid_start") else None,
        str(tuning.get("selection_valid_end")) if tuning.get("selection_valid_end") else None,
    )


def _frame_date(frame: pd.DataFrame, *, first: bool) -> str | None:
    """返回数据框首尾时间，供 summary 落盘。"""

    if frame.empty:
        return None
    series = pd.to_datetime(frame["timestamp"])
    value = series.min() if first else series.max()
    return pd.Timestamp(value).date().isoformat()


def _build_walk_forward_config(config: dict[str, object]):
    """把 JSON 结构的 walk-forward 配置还原成运行时对象。"""

    from ml.validation import WalkForwardConfig

    return WalkForwardConfig(
        train_size=int(config["train_size"]),
        valid_size=int(config["valid_size"]),
        step_size=int(config["step_size"]) if config.get("step_size") is not None else None,
        expanding=bool(config.get("expanding", True)),
        purge_size=int(config.get("purge_size", 0)),
        embargo_size=int(config.get("embargo_size", 0)),
    )
