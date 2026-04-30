from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import pandas as pd

from ml.features import build_technical_features, build_technical_features_with_diagnostics
from ml.labels import (
    add_cross_sectional_rank_label,
    add_future_return_label,
    future_rank_label_name,
    future_return_label_name,
)


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    diagnostics: dict[str, object] = field(default_factory=dict)


SUPPORTED_FEATURE_NORMALIZATION = ("none", "cross_sectional_zscore", "cross_sectional_rank")


def build_training_dataset(
    data: pd.DataFrame,
    label_horizon: int = 5,
    feature_columns: list[str] | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    target_mode: str = "future_return",
    feature_normalization: str = "none",
    factor_max_workers: int | None = None,
) -> DatasetBundle:
    selected_columns = list(feature_columns or [])
    if not selected_columns:
        raise ValueError("feature_columns must not be empty.")

    feature_started_at = monotonic()
    feature_result = build_technical_features_with_diagnostics(
        data=data,
        factor_names=selected_columns,
        reference_root=reference_root,
        market=market,
        max_workers=factor_max_workers,
    )
    feature_seconds = monotonic() - feature_started_at

    normalization_started_at = monotonic()
    features = apply_feature_normalization(
        frame=feature_result.frame,
        feature_columns=selected_columns,
        normalization=feature_normalization,
    )
    normalization_seconds = monotonic() - normalization_started_at

    label_started_at = monotonic()
    if target_mode == "future_return":
        labeled = add_future_return_label(features, horizon=label_horizon)
        label_column = future_return_label_name(label_horizon)
    elif target_mode == "cross_sectional_rank":
        labeled = add_cross_sectional_rank_label(features, horizon=label_horizon)
        label_column = future_rank_label_name(label_horizon)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    label_seconds = monotonic() - label_started_at

    diagnostics = {
        "feature_build_seconds": feature_seconds,
        "normalization_seconds": normalization_seconds,
        "label_build_seconds": label_seconds,
        "feature_builder": dict(feature_result.diagnostics),
    }
    return DatasetBundle(
        frame=labeled,
        feature_columns=selected_columns,
        label_column=label_column,
        diagnostics=diagnostics,
    )


def build_inference_dataset(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    feature_normalization: str = "none",
    factor_max_workers: int | None = None,
) -> pd.DataFrame:
    selected_columns = list(feature_columns or [])
    if not selected_columns:
        raise ValueError("feature_columns must not be empty.")
    features = build_technical_features(
        data=data,
        factor_names=selected_columns,
        reference_root=reference_root,
        market=market,
        max_workers=factor_max_workers,
    )
    features = apply_feature_normalization(
        frame=features,
        feature_columns=selected_columns,
        normalization=feature_normalization,
    )
    required = [column for column in selected_columns if column not in features.columns]
    if required:
        raise ValueError(f"Missing required feature columns: {required}")
    return features


def drop_rows_without_features(frame: pd.DataFrame, feature_columns: list[str], label_column: str | None = None) -> pd.DataFrame:
    columns = list(feature_columns)
    if label_column is not None:
        columns.append(label_column)
    return frame.dropna(subset=columns).reset_index(drop=True)


def apply_feature_normalization(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    normalization: str,
) -> pd.DataFrame:
    mode = str(normalization or "none")
    if mode not in SUPPORTED_FEATURE_NORMALIZATION:
        available = ", ".join(sorted(SUPPORTED_FEATURE_NORMALIZATION))
        raise ValueError(f"Unsupported feature_normalization: {mode}. Available values: {available}")
    if mode == "none" or frame.empty:
        return frame

    normalized = frame.copy()
    if mode == "cross_sectional_zscore":
        for column in feature_columns:
            means = normalized.groupby("timestamp")[column].transform("mean")
            stds = normalized.groupby("timestamp")[column].transform("std").replace(0.0, pd.NA)
            normalized[column] = (normalized[column] - means) / stds
        return normalized

    for column in feature_columns:
        normalized[column] = normalized.groupby("timestamp")[column].rank(method="average", pct=True)
    return normalized
