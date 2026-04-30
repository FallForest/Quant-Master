from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_load_experiment_spec_rejects_legacy_backtest_alias(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="signal_test"):
        load_experiment_spec(experiment_path)


def test_load_experiment_spec_reads_feature_normalization_and_signal_windows(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: feature_norm_demo",
                "features:",
                "  - return_5",
                "feature_normalization: cross_sectional_rank",
                "ic_decay_horizons: [1, 5, 10]",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-12-31",
                "signal_test:",
                "  name: full_oos",
                "  start_date: 2022-01-01",
                "  end_date: 2022-06-30",
                "signal_windows:",
                "  - name: h1",
                "    start_date: 2022-01-01",
                "    end_date: 2022-03-31",
                "  - name: h2",
                "    start_date: 2022-04-01",
                "    end_date: 2022-06-30",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(experiment_path)

    assert spec.feature_normalization == "cross_sectional_rank"
    assert spec.ic_decay_horizons == [1, 5, 10]
    assert spec.signal_test.name == "full_oos"
    assert [item.name for item in spec.signal_windows] == ["h1", "h2"]


def test_load_experiment_spec_applies_research_profile_defaults(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "name: profile_demo",
                "research_profile: hs300",
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

    assert spec.research_profile == "hs300"
    assert spec.market == "ashare"
    assert spec.provider == "parquet"
    assert spec.universe == "hs300"
    assert spec.benchmark_symbol == "sh000300"
    assert spec.baseline_manifest_path == "configs/experiments/official/hs300_official_baseline_manifest.json"


def test_load_experiment_group_spec_applies_profile_baseline_manifest(tmp_path: Path) -> None:
    group_path = tmp_path / "group.yaml"
    group_path.write_text(
        "\n".join(
            [
                "name: profile_group",
                "research_profile: zz500",
                "experiments:",
                "  - ../official/zz500_official_baseline.yaml",
            ]
        ),
        encoding="utf-8",
    )

    from ml.experiments.loader import load_experiment_group_spec

    spec = load_experiment_group_spec(group_path)

    assert spec.research_profile == "zz500"
    assert spec.baseline_manifest_path == "configs/experiments/official/zz500_official_baseline_manifest.json"
