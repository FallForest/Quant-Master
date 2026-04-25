from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.models import ValidationMetrics, aggregate_validation_metrics, evaluate_model, fit_model


@dataclass(slots=True)
class WalkForwardConfig:
    train_size: int
    valid_size: int
    step_size: int | None = None
    expanding: bool = True
    purge_size: int = 0
    embargo_size: int = 0


@dataclass(slots=True)
class ValidationSummary:
    mode: str
    metrics: ValidationMetrics
    folds: list[dict[str, object]]
    fold_count: int
    train_rows: int
    valid_rows: int


def split_dataset_by_time(
    frame: pd.DataFrame,
    train_end_date: str | None,
    valid_start_date: str | None,
    valid_end_date: str | None,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_end_date:
        return frame.copy(), pd.DataFrame(columns=frame.columns)

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    unique_dates = sorted(pd.to_datetime(ordered["timestamp"]).dropna().unique())
    train_end = pd.Timestamp(train_end_date)
    train_cutoff_index = _date_index_at_or_before(unique_dates, train_end)
    if train_cutoff_index is None:
        raise ValueError("Training split is empty. Adjust train_end_date.")

    purged_train_end_index = train_cutoff_index - max(0, int(purge_size))
    if purged_train_end_index < 0:
        raise ValueError("Training split is empty after purge. Reduce purge_size or expand the training window.")

    train_dates = set(unique_dates[: purged_train_end_index + 1])
    train_frame = ordered[ordered["timestamp"].isin(train_dates)].copy()

    valid_start_index = train_cutoff_index + 1 + max(0, int(embargo_size))
    if valid_start_index >= len(unique_dates):
        valid_frame = pd.DataFrame(columns=ordered.columns)
    else:
        valid_dates = unique_dates[valid_start_index:]
        valid_frame = ordered[ordered["timestamp"].isin(valid_dates)].copy()
    if valid_start_date:
        valid_frame = valid_frame[valid_frame["timestamp"] >= pd.Timestamp(valid_start_date)].copy()
    if valid_end_date:
        valid_frame = valid_frame[valid_frame["timestamp"] <= pd.Timestamp(valid_end_date)].copy()
    if train_frame.empty:
        raise ValueError("Training split is empty. Adjust train_end_date, purge_size, or embargo_size.")
    return train_frame.reset_index(drop=True), valid_frame.reset_index(drop=True)


def build_walk_forward_splits(frame: pd.DataFrame, config: WalkForwardConfig) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if config.train_size < 1 or config.valid_size < 1:
        raise ValueError("walk-forward train_size and valid_size must be positive integers.")
    if config.purge_size < 0 or config.embargo_size < 0:
        raise ValueError("walk-forward purge_size and embargo_size must be >= 0.")
    unique_dates = sorted(pd.to_datetime(frame["timestamp"]).dropna().unique())
    minimum_required = config.train_size + config.embargo_size + config.valid_size
    if len(unique_dates) < minimum_required:
        raise ValueError("Not enough unique timestamps for the requested walk-forward configuration.")

    step_size = config.step_size or config.valid_size
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end_index = config.train_size
    while train_end_index + config.valid_size <= len(unique_dates):
        train_start_index = 0 if config.expanding else train_end_index - config.train_size
        purged_train_end_index = train_end_index - config.purge_size
        valid_start_index = train_end_index + config.embargo_size
        valid_end_index = valid_start_index + config.valid_size
        if purged_train_end_index <= train_start_index or valid_end_index > len(unique_dates):
            train_end_index += step_size
            continue

        train_dates = unique_dates[train_start_index:purged_train_end_index]
        valid_dates = unique_dates[valid_start_index:valid_end_index]
        train_frame = frame[frame["timestamp"].isin(train_dates)].copy().reset_index(drop=True)
        valid_frame = frame[frame["timestamp"].isin(valid_dates)].copy().reset_index(drop=True)
        if not train_frame.empty and not valid_frame.empty:
            splits.append((train_frame, valid_frame))
        train_end_index += step_size
    if not splits:
        raise ValueError("Walk-forward configuration produced no usable folds.")
    return splits


def evaluate_holdout_validation(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict | None,
    train_end_date: str | None,
    valid_start_date: str | None,
    valid_end_date: str | None,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> ValidationSummary:
    train_frame, valid_frame = split_dataset_by_time(
        frame=frame,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    return evaluate_explicit_split(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_columns=feature_columns,
        label_column=label_column,
        model_name=model_name,
        model_params=model_params,
        mode="holdout",
    )


def evaluate_explicit_split(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict | None,
    mode: str,
) -> ValidationSummary:
    estimator = fit_model(
        frame=train_frame,
        feature_columns=feature_columns,
        label_column=label_column,
        model_name=model_name,
        model_params=model_params,
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
        mode=mode,
        metrics=metrics,
        folds=folds,
        fold_count=1,
        train_rows=int(len(train_frame)),
        valid_rows=int(len(valid_frame)),
    )


def evaluate_walk_forward_validation(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict | None,
    config: WalkForwardConfig,
) -> ValidationSummary:
    fold_metrics: list[ValidationMetrics] = []
    folds: list[dict[str, object]] = []
    total_train_rows = 0
    total_valid_rows = 0
    for index, (train_frame, valid_frame) in enumerate(build_walk_forward_splits(frame, config), start=1):
        estimator = fit_model(
            frame=train_frame,
            feature_columns=feature_columns,
            label_column=label_column,
            model_name=model_name,
            model_params=model_params,
        )
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
    aggregated = aggregate_validation_metrics(fold_metrics)
    return ValidationSummary(
        mode="walk_forward",
        metrics=aggregated,
        folds=folds,
        fold_count=len(folds),
        train_rows=int(total_train_rows),
        valid_rows=int(total_valid_rows),
    )


def _frame_date(frame: pd.DataFrame, *, first: bool) -> str | None:
    if frame.empty:
        return None
    series = pd.to_datetime(frame["timestamp"])
    value = series.min() if first else series.max()
    return pd.Timestamp(value).date().isoformat()


def _date_index_at_or_before(unique_dates: list, target: pd.Timestamp) -> int | None:
    index: int | None = None
    normalized_target = pd.Timestamp(target)
    for idx, value in enumerate(unique_dates):
        if pd.Timestamp(value) <= normalized_target:
            index = idx
        else:
            break
    return index
