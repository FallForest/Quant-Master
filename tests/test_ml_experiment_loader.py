from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.loader import load_experiment_spec


def test_load_experiment_spec_reads_signal_test_block(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: signal_test_demo",
                "features:",
                "  - return_5",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-12-31",
                "signal_test:",
                "  start_date: 2022-01-01",
                "  end_date: 2022-06-30",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(experiment_path)

    assert spec.signal_test.start_date == "2022-01-01"
    assert spec.signal_test.end_date == "2022-06-30"
    assert spec.train.candidate_selection.metric == "oos_spearman_ic"


def test_load_experiment_spec_accepts_legacy_backtest_alias(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: legacy_alias_demo",
                "features:",
                "  - return_5",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-12-31",
                "backtest:",
                "  start_date: 2022-01-01",
                "  end_date: 2022-06-30",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(experiment_path)

    assert spec.signal_test.start_date == "2022-01-01"
    assert spec.signal_test.end_date == "2022-06-30"
