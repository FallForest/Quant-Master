from __future__ import annotations

import sys
import pytest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.dataset import apply_feature_normalization
from ml.diagnostics import build_ic_decay_profile, build_signal_slice_diagnostics
from ml.experiments.compare import build_comparison_frame
from ml.labels import add_cross_sectional_rank_label, future_rank_label_name, future_return_label_name
from ml.models import evaluate_model, evaluate_scored_frame


class DummyEstimator:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = predictions

    def predict(self, features) -> list[float]:
        return list(self._predictions[: len(features)])


def test_evaluate_model_uses_cross_sectional_ic_series() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "feature_a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    estimator = DummyEstimator(predictions=[1.0, 2.0, 3.0, 3.0, 2.0, 1.0])

    metrics = evaluate_model(
        estimator=estimator,
        frame=frame,
        feature_columns=["feature_a"],
        label_column="label",
    )

    assert metrics.ic == 0.0
    assert metrics.rank_ic == 0.0
    assert metrics.ic_std == 1.0
    assert metrics.rank_icir == 0.0
    assert metrics.as_dict()["ic_std"] == 1.0
    assert metrics.as_dict()["rank_ic"] == 0.0


def test_build_comparison_frame_exposes_qlib_style_signal_metrics() -> None:
    group_summary = {
        "experiments": [
            {
                "name": "demo",
                "group": "signal",
                "model": "ridge",
                "features": ["a", "b"],
                "training_metadata": {
                    "validation_mode": "walk_forward",
                    "target_mode": "cross_sectional_rank",
                    "tuning": {"enabled": False},
                },
                "validation_metrics": {
                    "ic": 0.03,
                    "rank_ic": 0.05,
                    "ic_std": 0.02,
                    "rank_ic_std": 0.03,
                    "icir": 1.5,
                    "rank_icir": 2.5,
                    "mae": 0.2,
                    "r2": 0.01,
                },
                "signal_test_metrics": {
                    "ic": 0.02,
                    "rank_ic": 0.04,
                    "ic_std": 0.01,
                    "rank_ic_std": 0.02,
                    "icir": 2.0,
                    "rank_icir": 4.0,
                },
                "research_diagnostics": {"pbo": 0.2, "dsr": 0.8},
                "candidate_selection": {},
                "report_dir": "reports/demo",
                "artifact_path": "artifacts/demo",
            }
        ]
    }

    frame = build_comparison_frame(group_summary)

    row = frame.iloc[0]
    assert row["ic"] == 0.03
    assert row["rank_ic"] == 0.05
    assert row["ic_std"] == 0.02
    assert row["rank_icir"] == 2.5
    assert row["test_ic"] == 0.02
    assert row["test_rank_ic"] == 0.04
    assert row["test_ic_std"] == 0.01
    assert row["test_rank_icir"] == 4.0


def test_apply_feature_normalization_cross_sectional_modes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "feature_a": [1.0, 3.0, 2.0, 4.0],
        }
    )

    zscore = apply_feature_normalization(
        frame=frame,
        feature_columns=["feature_a"],
        normalization="cross_sectional_zscore",
    )
    ranked = apply_feature_normalization(
        frame=frame,
        feature_columns=["feature_a"],
        normalization="cross_sectional_rank",
    )

    assert zscore.loc[0, "feature_a"] < 0
    assert zscore.loc[1, "feature_a"] > 0
    assert list(ranked["feature_a"]) == [0.5, 1.0, 0.5, 1.0]


def test_add_cross_sectional_rank_label_handles_empty_frame() -> None:
    frame = pd.DataFrame(columns=["timestamp", "symbol", "close"])

    labeled = add_cross_sectional_rank_label(frame, horizon=5)

    assert future_return_label_name(5) in labeled.columns
    assert future_rank_label_name(5) in labeled.columns
    assert labeled.empty


def test_build_ic_decay_profile_returns_requested_horizons() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["A", "B", "A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
            "feature_a": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        }
    )

    estimator = DummyEstimator(predictions=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    profile = build_ic_decay_profile(
        estimator=estimator,
        frame=frame,
        feature_columns=["feature_a"],
        target_mode="future_return",
        horizons=[1, 2],
    )

    assert [row["horizon"] for row in profile] == [1, 2]
    assert all("rank_ic" in row for row in profile)


def test_evaluate_scored_frame_matches_model_evaluation_shape() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "prediction": [0.9, 0.1, 0.8, 0.2],
            "label": [1.0, 0.0, 1.0, 0.0],
        }
    )

    metrics = evaluate_scored_frame(
        frame=frame,
        prediction_column="prediction",
        label_column="label",
    )

    assert metrics.rank_ic == pytest.approx(1.0)
    assert metrics.rank_icir == 0.0


def test_build_signal_slice_diagnostics_uses_industry_reference(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference" / "ashare"
    (reference_root / "industry").mkdir(parents=True)
    (reference_root / "index").mkdir(parents=True)
    (reference_root / "fundamentals").mkdir(parents=True)
    (reference_root / "industry" / "000001.csv").write_text(
        "symbol,change_date,standard,sector,industry_level_1,industry_level_2,industry_level_3,industry_code\n"
        "000001,2024-01-01,申银万国行业分类标准,金融,银行,股份制银行,,801780\n",
        encoding="utf-8",
    )
    (reference_root / "industry" / "000002.csv").write_text(
        "symbol,change_date,standard,sector,industry_level_1,industry_level_2,industry_level_3,industry_code\n"
        "000002,2024-01-01,申银万国行业分类标准,地产,房地产,住宅开发,,801180\n",
        encoding="utf-8",
    )
    (reference_root / "index" / "sh000300.csv").write_text(
        "timestamp,close\n2024-01-01,100\n2024-01-02,101\n2024-01-03,102\n2024-01-04,103\n",
        encoding="utf-8",
    )
    for symbol, share_capital in [("000001", 1000000), ("000002", 2000000)]:
        (reference_root / "fundamentals" / f"{symbol}.csv").write_text(
            "symbol,report_date,available_date,total_assets,total_parent_equity,share_capital,total_operate_income_ttm,operate_cost_ttm,operate_profit_ttm,parent_netprofit_ttm,netcash_operate_ttm,asset_growth,investment_to_assets,accruals\n"
            f"{symbol},2023-12-31,2023-12-31,1,1,{share_capital},1,1,1,1,1,0,0,0\n",
            encoding="utf-8",
        )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["000001", "000002", "000001", "000002"],
            "open": [10.0, 20.0, 10.5, 20.5],
            "high": [10.2, 20.4, 10.7, 20.8],
            "low": [9.9, 19.8, 10.1, 20.2],
            "close": [10.1, 20.1, 10.6, 20.6],
            "volume": [1000, 1200, 1100, 1300],
            "feature_a": [1.0, 0.0, 1.0, 0.0],
            "label": [1.0, 0.0, 1.0, 0.0],
        }
    )

    diagnostics = build_signal_slice_diagnostics(
        estimator=DummyEstimator(predictions=[1.0, 0.0, 1.0, 0.0]),
        frame=frame,
        feature_columns=["feature_a"],
        label_column="label",
        reference_root=str(tmp_path / "reference"),
        market="ashare",
        benchmark_symbol="sh000300",
    )

    assert diagnostics["industry_buckets"]
    assert any(item["name"] == "银行" for item in diagnostics["industry_buckets"])
