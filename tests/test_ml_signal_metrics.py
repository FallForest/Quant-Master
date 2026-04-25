from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.compare import build_comparison_frame
from ml.models import evaluate_model


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

    assert metrics.pearson_ic == 0.0
    assert metrics.spearman_ic == 0.0
    assert metrics.ic_std == 1.0
    assert metrics.ic_ir == 0.0
    assert metrics.as_dict()["oos_ic_std"] == 1.0
    assert metrics.as_dict()["oos_spearman_ic"] == 0.0


def test_build_comparison_frame_exposes_oos_signal_metrics() -> None:
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
                    "oos_pearson_ic": 0.03,
                    "oos_spearman_ic": 0.05,
                    "oos_ic_std": 0.02,
                    "oos_ic_ir": 2.5,
                    "oos_ndcg_at_10": 0.51,
                    "mae": 0.2,
                    "r2": 0.01,
                },
                "signal_test_metrics": {
                    "oos_pearson_ic": 0.02,
                    "oos_spearman_ic": 0.04,
                    "oos_ic_std": 0.01,
                    "oos_ic_ir": 4.0,
                    "oos_ndcg_at_10": 0.49,
                },
                "signal_test_rows": 128,
                "research_diagnostics": {"pbo": 0.2, "dsr": 0.8},
                "candidate_selection": {},
                "report_dir": "reports/demo",
                "artifact_path": "artifacts/demo",
            }
        ]
    }

    frame = build_comparison_frame(group_summary)

    row = frame.iloc[0]
    assert row["oos_pearson_ic"] == 0.03
    assert row["oos_spearman_ic"] == 0.05
    assert row["oos_ic_std"] == 0.02
    assert row["oos_ic_ir"] == 2.5
    assert row["pearson_ic"] == 0.03
    assert row["spearman_ic"] == 0.05
    assert row["test_pearson_ic"] == 0.02
    assert row["test_spearman_ic"] == 0.04
    assert row["test_ic_std"] == 0.01
    assert row["test_ic_ir"] == 4.0
    assert row["signal_test_rows"] == 128
