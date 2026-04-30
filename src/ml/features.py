from __future__ import annotations

"""Feature-construction entry points for ML datasets."""

from dataclasses import dataclass

import pandas as pd

from ml.factors.builder import FactorBuildResult, build_factor_frame_with_diagnostics


@dataclass(slots=True)
class TechnicalFeatureBuildResult:
    frame: pd.DataFrame
    diagnostics: dict[str, object]


def build_technical_features(
    data: pd.DataFrame,
    *,
    factor_names: list[str],
    params: dict | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    max_workers: int | None = None,
) -> pd.DataFrame:
    return build_technical_features_with_diagnostics(
        data=data,
        factor_names=factor_names,
        params=params,
        reference_root=reference_root,
        market=market,
        max_workers=max_workers,
    ).frame


def build_technical_features_with_diagnostics(
    data: pd.DataFrame,
    *,
    factor_names: list[str],
    params: dict | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    max_workers: int | None = None,
) -> TechnicalFeatureBuildResult:
    if params:
        raise ValueError("feature_params has been removed. Use explicit factor_names instead.")
    if not factor_names:
        raise ValueError("At least one factor name is required.")

    result: FactorBuildResult = build_factor_frame_with_diagnostics(
        data=data,
        factor_names=list(factor_names),
        reference_root=reference_root,
        market=market,
        max_workers=max_workers,
    )
    return TechnicalFeatureBuildResult(frame=result.frame, diagnostics=result.diagnostics)
