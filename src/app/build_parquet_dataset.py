from __future__ import annotations

from datetime import datetime
from pathlib import Path

from data.providers.csv_provider import CsvDataProvider
from data.io_parallel import format_adaptive_io_failures, run_adaptive_io_tasks
from data.providers.parquet_provider import ParquetDataProvider


def build_parquet_dataset(
    market: str,
    input_root: str,
    output_root: str,
    universe_root: str,
    timeframe: str,
    adjust: str,
    symbols: list[str] | None = None,
    universe: str | None = None,
    max_workers: int | None = None,
) -> list[str]:
    source = CsvDataProvider(
        base_path=input_root,
        universe_root=universe_root,
        default_adjust=adjust,
    )
    target = ParquetDataProvider(
        base_path=output_root,
        universe_root=universe_root,
        default_adjust=adjust,
    )

    resolved_symbols = [source.normalize_symbol(symbol) for symbol in symbols] if symbols else []
    if not resolved_symbols:
        resolved_symbols = source.load_universe(market=market, universe=universe, date=None)

    if not resolved_symbols:
        raise ValueError("No symbols found to convert. Provide --symbols, --universe, or prepare CSV data first.")

    def _build_one_symbol(symbol: str) -> str | None:
        bars = source.load_bars(
            symbol=symbol,
            market=market,
            start=datetime(1900, 1, 1),
            end=datetime(2100, 1, 1),
            timeframe=timeframe,
        )
        if bars.empty:
            return None

        target_path = target.resolve_data_path(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            suffix=target.file_suffix,
        )
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            bars.to_parquet(target_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Building parquet datasets requires `pyarrow`. Run `py -m pip install -r requirements.txt` first."
            ) from exc
        return str(target_path)

    execution = run_adaptive_io_tasks(
        items=resolved_symbols,
        worker_fn=_build_one_symbol,
        progress_desc="Building parquet dataset",
        progress_unit="symbol",
        max_workers=max_workers,
    )
    failures = [outcome for outcome in execution.outcomes if not outcome.succeeded]
    if failures:
        examples = format_adaptive_io_failures(failures)
        raise RuntimeError(f"Failed to build parquet files for {len(failures)} symbol(s). Examples: {examples}")

    paths = [outcome.value for outcome in execution.outcomes if outcome.value]
    return paths
