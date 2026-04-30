from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pandas as pd

from data.io_parallel import AdaptiveIoOutcome, AdaptiveIoReport, format_adaptive_io_failures, run_adaptive_io_tasks


DEFAULT_EMPTY_BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


@dataclass(slots=True)
class LoadedBarsBatch:
    frame: pd.DataFrame
    report: AdaptiveIoReport[str, pd.DataFrame]


def load_bars_for_symbols(
    *,
    provider,
    market: str,
    timeframe: str,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    progress_desc: str,
    show_progress: bool = True,
    max_workers: int | None = None,
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    return load_bars_for_symbols_batch(
        provider=provider,
        market=market,
        timeframe=timeframe,
        symbols=symbols,
        start=start,
        end=end,
        progress_desc=progress_desc,
        show_progress=show_progress,
        max_workers=max_workers,
        empty_columns=empty_columns,
    ).frame


def load_bars_for_symbols_batch(
    *,
    provider,
    market: str,
    timeframe: str,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    progress_desc: str,
    show_progress: bool = True,
    max_workers: int | None = None,
    empty_columns: list[str] | None = None,
) -> LoadedBarsBatch:
    ordered_symbols = list(symbols)

    def _load_one_symbol(symbol: str) -> pd.DataFrame:
        return provider.load_bars(symbol=symbol, market=market, start=start, end=end, timeframe=timeframe)

    execution = run_adaptive_io_tasks(
        items=ordered_symbols,
        worker_fn=_load_one_symbol,
        progress_desc=progress_desc,
        progress_unit="symbol",
        show_progress=show_progress,
        max_workers=max_workers,
    )

    failures = [outcome for outcome in execution.outcomes if not outcome.succeeded]
    if failures:
        raise RuntimeError(_format_load_failure_message(failures))

    frames = [outcome.value for outcome in execution.outcomes if outcome.value is not None and not outcome.value.empty]
    if not frames:
        return LoadedBarsBatch(
            frame=pd.DataFrame(columns=empty_columns or DEFAULT_EMPTY_BAR_COLUMNS),
            report=execution,
        )
    return LoadedBarsBatch(
        frame=pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True),
        report=execution,
    )


def _format_load_failure_message(failures: list[AdaptiveIoOutcome[str, pd.DataFrame]], *, max_examples: int = 5) -> str:
    examples = format_adaptive_io_failures(failures, max_examples=max_examples)
    return f"Failed to load bars for {len(failures)} symbol(s). Examples: {examples}"
