from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.diagnostics import build_signal_slice_diagnostics


class DummyEstimator:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = predictions

    def predict(self, features) -> list[float]:
        return list(self._predictions[: len(features)])


def test_build_signal_slice_diagnostics_reuses_existing_industry_and_market_cap_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["000001", "000002", "000001", "000002"],
            "open": [10.0, 20.0, 10.5, 20.5],
            "high": [10.2, 20.4, 10.7, 20.8],
            "low": [9.9, 19.8, 10.1, 20.2],
            "close": [10.1, 20.1, 10.6, 20.6],
            "volume": [1000, 1200, 1100, 1300],
            "industry_level_1": ["银行", "地产", "银行", "地产"],
            "log_total_mkt_cap": [10.0, 12.0, 10.2, 12.2],
            "feature_a": [1.0, 0.0, 1.0, 0.0],
            "label": [1.0, 0.0, 1.0, 0.0],
        }
    )

    diagnostics = build_signal_slice_diagnostics(
        estimator=DummyEstimator(predictions=[1.0, 0.0, 1.0, 0.0]),
        frame=frame,
        feature_columns=["feature_a"],
        label_column="label",
        reference_root="missing_reference_root_ok_for_pre_enriched_frame",
        market="ashare",
        benchmark_symbol="sh000300",
    )

    assert diagnostics["industry_buckets"]
    assert diagnostics["market_cap_buckets"]
