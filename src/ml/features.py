from __future__ import annotations

import pandas as pd

from ml.factors.builder import build_factor_frame


def build_technical_features(
    data: pd.DataFrame,
    *,
    factor_names: list[str],
    params: dict | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
) -> pd.DataFrame:
    if params:
        raise ValueError("feature_params has been removed. Use explicit factor_names instead.")
    if not factor_names:
        raise ValueError("At least one factor name is required.")
    return build_factor_frame(
        data=data,
        factor_names=list(factor_names),
        reference_root=reference_root,
        market=market,
    )
