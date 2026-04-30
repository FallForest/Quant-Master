from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

import pandas as pd

from data.loading import load_bars_for_symbols_batch
from data.provider_factory import build_data_provider
from ml.dataset import DatasetBundle, build_training_dataset, drop_rows_without_features


@dataclass(slots=True)
class PreparedSignalDataset:
    frame: object
    feature_columns: list[str]
    label_column: str
    symbols: list[str]
    diagnostics: dict[str, object] = field(default_factory=dict)


class SignalDatasetCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._datasets: dict[tuple[object, ...], PreparedSignalDataset] = {}

    def get(self, key: tuple[object, ...]) -> PreparedSignalDataset | None:
        with self._lock:
            return self._datasets.get(key)

    def put(self, key: tuple[object, ...], dataset: PreparedSignalDataset) -> PreparedSignalDataset:
        with self._lock:
            self._datasets[key] = dataset
        return dataset


def prepare_signal_dataset(
    *,
    market: str,
    provider_name: str,
    data_root: str,
    universe_root: str,
    reference_root: str,
    adjust: str,
    timeframe: str,
    symbols: list[str] | None,
    universe: str | None,
    start_date: str,
    end_date: str,
    feature_columns: list[str],
    feature_normalization: str,
    label_horizon: int,
    target_mode: str,
    progress_desc: str,
    dataset_cache: SignalDatasetCache | None = None,
    history_padding_days: int = 0,
    trim_to_requested_range: bool = False,
    bar_max_workers: int | None = None,
    factor_max_workers: int | None = None,
    show_progress: bool = True,
) -> PreparedSignalDataset:
    selected_features = list(feature_columns or [])
    if not selected_features:
        raise ValueError("feature_columns must not be empty.")

    cache_key = (
        str(market),
        str(provider_name),
        str(data_root),
        str(universe_root),
        str(reference_root),
        str(adjust),
        str(timeframe),
        tuple(str(item) for item in (symbols or [])),
        str(universe) if universe is not None else None,
        str(start_date),
        str(end_date),
        tuple(selected_features),
        str(feature_normalization),
        int(label_horizon),
        str(target_mode),
        int(history_padding_days),
        bool(trim_to_requested_range),
        int(bar_max_workers) if bar_max_workers is not None else None,
        int(factor_max_workers) if factor_max_workers is not None else None,
    )
    if dataset_cache is not None:
        cached = dataset_cache.get(cache_key)
        if cached is not None:
            return cached

    total_started_at = monotonic()
    provider = build_data_provider(
        provider_name=provider_name,
        data_root=data_root,
        universe_root=universe_root,
        adjust=adjust,
    )
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    start = requested_start - pd.offsets.BDay(max(0, int(history_padding_days)))
    end = requested_end
    resolved_symbols = list(symbols or [])
    if not resolved_symbols:
        resolved_symbols = provider.load_universe(market=market, universe=universe, date=start.to_pydatetime())
    if not resolved_symbols:
        raise ValueError("No symbols were resolved for ML training.")

    bar_started_at = monotonic()
    bar_batch = load_bars_for_symbols_batch(
        provider=provider,
        market=market,
        timeframe=timeframe,
        symbols=resolved_symbols,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        progress_desc=progress_desc,
        show_progress=show_progress,
        max_workers=bar_max_workers,
        empty_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )
    bar_load_seconds = monotonic() - bar_started_at

    dataset_started_at = monotonic()
    bundle: DatasetBundle = build_training_dataset(
        data=bar_batch.frame,
        label_horizon=label_horizon,
        feature_columns=selected_features,
        reference_root=reference_root,
        market=market,
        target_mode=target_mode,
        feature_normalization=feature_normalization,
        factor_max_workers=factor_max_workers,
    )
    dataset_build_seconds = monotonic() - dataset_started_at

    filter_started_at = monotonic()
    prepared_frame = drop_rows_without_features(
        frame=bundle.frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
    )
    filter_seconds = monotonic() - filter_started_at

    trim_seconds = 0.0
    if trim_to_requested_range and not prepared_frame.empty:
        trim_started_at = monotonic()
        timestamps = pd.to_datetime(prepared_frame["timestamp"], errors="coerce")
        prepared_frame = prepared_frame.loc[
            timestamps.ge(requested_start) & timestamps.le(requested_end)
        ].reset_index(drop=True)
        trim_seconds = monotonic() - trim_started_at

    io_report = bar_batch.report
    io_resolved_workers = max(
        [int(io_report.initial_workers)]
        + [int(item.workers) for item in io_report.telemetry]
    )
    diagnostics = {
        "symbol_count": len(resolved_symbols),
        "input_rows": int(len(bar_batch.frame)),
        "prepared_rows": int(len(prepared_frame)),
        "bar_load_seconds": bar_load_seconds,
        "dataset_build_seconds": dataset_build_seconds,
        "filter_seconds": filter_seconds,
        "trim_seconds": trim_seconds,
        "elapsed_seconds": monotonic() - total_started_at,
        "bar_loader": {
            "requested_workers": int(bar_max_workers) if bar_max_workers is not None else None,
            "planned_max_workers": int(io_report.planned_max_workers),
            "initial_workers": int(io_report.initial_workers),
            "resolved_workers": io_resolved_workers,
            "telemetry_batches": len(io_report.telemetry),
        },
        "dataset_builder": dict(bundle.diagnostics),
    }
    prepared = PreparedSignalDataset(
        frame=prepared_frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
        symbols=resolved_symbols,
        diagnostics=diagnostics,
    )
    if dataset_cache is not None:
        return dataset_cache.put(cache_key, prepared)
    return prepared
