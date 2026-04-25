from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_return_label(
    data: pd.DataFrame,
    horizon: int = 5,
    price_column: str = "close",
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if data.empty:
        return data.copy()

    ordered = data.sort_values(["symbol", "timestamp"]).copy()
    label_column = future_return_label_name(horizon)
    ordered[label_column] = (
        ordered.groupby("symbol")[price_column].shift(-horizon) / ordered[price_column] - 1.0
    )
    ordered[label_column] = ordered[label_column].replace([np.inf, -np.inf], np.nan)
    return ordered


def add_cross_sectional_rank_label(
    data: pd.DataFrame,
    horizon: int = 5,
    price_column: str = "close",
) -> pd.DataFrame:
    labeled = add_future_return_label(data=data, horizon=horizon, price_column=price_column)
    future_column = future_return_label_name(horizon)
    rank_column = future_rank_label_name(horizon)
    labeled[rank_column] = labeled.groupby("timestamp")[future_column].rank(method="average", pct=True)
    labeled[rank_column] = labeled[rank_column].replace([np.inf, -np.inf], np.nan)
    return labeled


def future_return_label_name(horizon: int) -> str:
    return f"future_return_{int(horizon)}d"


def future_rank_label_name(horizon: int) -> str:
    return f"future_return_rank_{int(horizon)}d"
