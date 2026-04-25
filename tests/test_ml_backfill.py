from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.backfill import backfill_signal_metrics_from_path
from ml.training import train_ml_signal_model


def test_backfill_signal_metrics_from_path(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact"
    report_dir = tmp_path / "report"
    experiment_path = tmp_path / "experiment.yaml"

    train_ml_signal_model(
        market="ashare",
        provider_name="csv",
        data_root="data/raw",
        universe_root="data/universe",
        reference_root="data/reference",
        adjust="qfq",
        timeframe="1d",
        symbols=["000001", "000002", "000063", "000066"],
        universe=None,
        start_date="2021-01-01",
        end_date="2021-06-30",
        artifact_path=str(artifact_path),
        model_name="ridge",
        model_params=None,
        feature_columns=["return_5", "volatility_5", "volume_ratio_5"],
        label_horizon=5,
        target_mode="future_return",
        train_end_date="2021-04-30",
        valid_start_date="2021-05-01",
        valid_end_date="2021-06-30",
        validation_mode="holdout",
        walk_forward_config=None,
        tuning_config=None,
        purge_size=0,
        embargo_size=0,
    )

    experiment_path.write_text(
        "\n".join(
            [
                "name: test_backfill",
                "market: ashare",
                "provider: csv",
                "data_root: data/raw",
                "universe_root: data/universe",
                "reference_root: data/reference",
                "adjust: qfq",
                "timeframe: 1d",
                "symbols:",
                "  - '000001'",
                "  - '000002'",
                "  - '000063'",
                "  - '000066'",
                "features:",
                "  - return_5",
                "  - volatility_5",
                "  - volume_ratio_5",
                "model: ridge",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-06-30",
                "  validation_mode: holdout",
                "  train_end_date: 2021-04-30",
                "  valid_start_date: 2021-05-01",
                "  valid_end_date: 2021-06-30",
                "  label_horizon: 5",
                "  target_mode: future_return",
                "signal_test:",
                "  start_date: 2021-07-01",
                "  end_date: 2021-07-31",
                "report:",
                f"  output_dir: {report_dir.as_posix()}",
                f"  artifact_path: {artifact_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    summary = backfill_signal_metrics_from_path(experiment_path)

    assert "oos_spearman_ic" in summary["validation_metrics"]
    report_summary = json.loads((report_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    assert "oos_ic_std" in report_summary["validation_metrics"]
