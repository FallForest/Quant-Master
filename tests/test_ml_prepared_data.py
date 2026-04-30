from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.dataset import DatasetBundle
from ml.factors import estimate_factor_history_lookback
from ml.prepared_data import SignalDatasetCache, prepare_signal_dataset
from data.io_parallel import AdaptiveIoReport
from data.loading import LoadedBarsBatch


def test_prepare_signal_dataset_uses_cache(monkeypatch) -> None:
    calls = {"provider": 0, "universe": 0, "bars": 0, "dataset": 0}

    class FakeProvider:
        def load_universe(self, *, market: str, universe: str | None, date) -> list[str]:
            calls["universe"] += 1
            return ["000001"]

    def fake_build_provider(**kwargs):
        calls["provider"] += 1
        return FakeProvider()

    def fake_load_bars_for_symbols_batch(**kwargs):
        calls["bars"] += 1
        return LoadedBarsBatch(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2021-01-04", "2021-01-05"]),
                    "symbol": ["000001", "000001"],
                    "close": [10.0, 10.2],
                }
            ),
            report=AdaptiveIoReport(outcomes=[], telemetry=[], planned_max_workers=1, initial_workers=1),
        )

    def fake_build_training_dataset(**kwargs):
        calls["dataset"] += 1
        return DatasetBundle(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2021-01-04", "2021-01-05"]),
                    "symbol": ["000001", "000001"],
                    "return_5": [0.1, 0.2],
                    "future_return_5": [0.3, 0.4],
                }
            ),
            feature_columns=["return_5"],
            label_column="future_return_5",
        )

    monkeypatch.setattr("ml.prepared_data.build_data_provider", fake_build_provider)
    monkeypatch.setattr("ml.prepared_data.load_bars_for_symbols_batch", fake_load_bars_for_symbols_batch)
    monkeypatch.setattr("ml.prepared_data.build_training_dataset", fake_build_training_dataset)

    cache = SignalDatasetCache()
    first = prepare_signal_dataset(
        market="ashare",
        provider_name="parquet",
        data_root="data/lake",
        universe_root="data/universe",
        reference_root="data/reference",
        adjust="qfq",
        timeframe="1d",
        symbols=None,
        universe="hs300",
        start_date="2021-01-01",
        end_date="2021-01-31",
        feature_columns=["return_5"],
        feature_normalization="none",
        label_horizon=5,
        target_mode="future_return",
        progress_desc="cache test",
        dataset_cache=cache,
    )
    second = prepare_signal_dataset(
        market="ashare",
        provider_name="parquet",
        data_root="data/lake",
        universe_root="data/universe",
        reference_root="data/reference",
        adjust="qfq",
        timeframe="1d",
        symbols=None,
        universe="hs300",
        start_date="2021-01-01",
        end_date="2021-01-31",
        feature_columns=["return_5"],
        feature_normalization="none",
        label_horizon=5,
        target_mode="future_return",
        progress_desc="cache test",
        dataset_cache=cache,
    )

    assert first is second
    assert calls == {"provider": 1, "universe": 1, "bars": 1, "dataset": 1}


def test_prepare_signal_dataset_supports_history_padding_and_trim(monkeypatch) -> None:
    observed = {}

    class FakeProvider:
        def load_universe(self, *, market: str, universe: str | None, date) -> list[str]:
            observed["universe_date"] = pd.Timestamp(date)
            return ["000001"]

    def fake_build_provider(**kwargs):
        return FakeProvider()

    def fake_load_bars_for_symbols_batch(**kwargs):
        observed["bars_start"] = pd.Timestamp(kwargs["start"])
        observed["bars_end"] = pd.Timestamp(kwargs["end"])
        return LoadedBarsBatch(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"]),
                    "symbol": ["000001", "000001", "000001"],
                    "close": [10.0, 10.2, 10.3],
                }
            ),
            report=AdaptiveIoReport(outcomes=[], telemetry=[], planned_max_workers=1, initial_workers=1),
        )

    def fake_build_training_dataset(**kwargs):
        return DatasetBundle(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"]),
                    "symbol": ["000001", "000001", "000001"],
                    "momentum_252_21": [0.1, 0.2, 0.3],
                    "future_return_5": [0.3, 0.4, 0.5],
                }
            ),
            feature_columns=["momentum_252_21"],
            label_column="future_return_5",
        )

    monkeypatch.setattr("ml.prepared_data.build_data_provider", fake_build_provider)
    monkeypatch.setattr("ml.prepared_data.load_bars_for_symbols_batch", fake_load_bars_for_symbols_batch)
    monkeypatch.setattr("ml.prepared_data.build_training_dataset", fake_build_training_dataset)

    prepared = prepare_signal_dataset(
        market="ashare",
        provider_name="parquet",
        data_root="data/lake",
        universe_root="data/universe",
        reference_root="data/reference",
        adjust="qfq",
        timeframe="1d",
        symbols=None,
        universe="hs300",
        start_date="2025-01-01",
        end_date="2025-01-31",
        feature_columns=["momentum_252_21"],
        feature_normalization="none",
        label_horizon=5,
        target_mode="future_return",
        progress_desc="padding test",
        history_padding_days=252,
        trim_to_requested_range=True,
    )

    assert observed["bars_start"] < pd.Timestamp("2025-01-01")
    assert observed["bars_end"] == pd.Timestamp("2025-01-31")
    assert observed["universe_date"] == observed["bars_start"]
    assert prepared.frame["timestamp"].min() == pd.Timestamp("2025-01-02")
    assert prepared.frame["timestamp"].max() == pd.Timestamp("2025-01-03")


def test_prepare_signal_dataset_records_parallel_diagnostics(monkeypatch) -> None:
    class FakeProvider:
        def load_universe(self, *, market: str, universe: str | None, date) -> list[str]:
            return ["000001", "000002"]

    def fake_build_provider(**kwargs):
        return FakeProvider()

    def fake_load_bars_for_symbols_batch(**kwargs):
        return LoadedBarsBatch(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                    "symbol": ["000001", "000002"],
                    "close": [10.0, 10.2],
                }
            ),
            report=AdaptiveIoReport(outcomes=[], telemetry=[], planned_max_workers=4, initial_workers=2),
        )

    def fake_build_training_dataset(**kwargs):
        return DatasetBundle(
            frame=pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                    "symbol": ["000001", "000002"],
                    "return_5": [0.1, 0.2],
                    "future_return_5": [0.3, 0.4],
                }
            ),
            feature_columns=["return_5"],
            label_column="future_return_5",
            diagnostics={"feature_builder": {"resolved_workers": 2}},
        )

    monkeypatch.setattr("ml.prepared_data.build_data_provider", fake_build_provider)
    monkeypatch.setattr("ml.prepared_data.load_bars_for_symbols_batch", fake_load_bars_for_symbols_batch)
    monkeypatch.setattr("ml.prepared_data.build_training_dataset", fake_build_training_dataset)

    prepared = prepare_signal_dataset(
        market="ashare",
        provider_name="parquet",
        data_root="data/lake",
        universe_root="data/universe",
        reference_root="data/reference",
        adjust="qfq",
        timeframe="1d",
        symbols=None,
        universe="hs300",
        start_date="2025-01-01",
        end_date="2025-01-31",
        feature_columns=["return_5"],
        feature_normalization="none",
        label_horizon=5,
        target_mode="future_return",
        progress_desc="diagnostics test",
        bar_max_workers=4,
        factor_max_workers=2,
        show_progress=False,
    )

    assert prepared.diagnostics["bar_loader"]["planned_max_workers"] == 4
    assert prepared.diagnostics["bar_loader"]["resolved_workers"] == 2
    assert prepared.diagnostics["dataset_builder"]["feature_builder"]["resolved_workers"] == 2


def test_estimate_factor_history_lookback_handles_long_window_factors() -> None:
    assert estimate_factor_history_lookback(["return_20", "close_location_value"]) == 20
    assert estimate_factor_history_lookback(["beta_120_hs300"]) == 121
    assert estimate_factor_history_lookback(["momentum_252_21"]) == 252
