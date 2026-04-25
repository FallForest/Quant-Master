from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.features import build_technical_features
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


def build_training_dataset(
    data: pd.DataFrame,
    label_horizon: int = 5,
    feature_columns: list[str] | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    target_mode: str = "future_return",
) -> DatasetBundle:
    selected_columns = list(feature_columns or [])
    if not selected_columns:
        raise ValueError("feature_columns must not be empty.")
    features = build_technical_features(
        data=data,
        factor_names=selected_columns,
        reference_root=reference_root,
        market=market,
    )
    if target_mode == "future_return":
        labeled = add_future_return_label(features, horizon=label_horizon)
        label_column = future_return_label_name(label_horizon)
    elif target_mode == "cross_sectional_rank":
        labeled = add_cross_sectional_rank_label(features, horizon=label_horizon)
        label_column = future_rank_label_name(label_horizon)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return DatasetBundle(
        frame=labeled,
        feature_columns=selected_columns,
        label_column=label_column,
    )


def build_inference_dataset(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
) -> pd.DataFrame:
    selected_columns = list(feature_columns or [])
    if not selected_columns:
        raise ValueError("feature_columns must not be empty.")
    features = build_technical_features(
        data=data,
        factor_names=selected_columns,
        reference_root=reference_root,
        market=market,
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
