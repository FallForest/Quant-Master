from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
import pandas as pd

from data.providers.akshare_provider import AKShareAshareProvider
from data.pipeline_tracking import PipelineRunTracker
from data.providers.csv_provider import CsvDataProvider
from data.io_parallel import run_adaptive_io_tasks
from data.sync_metadata import csv_target_is_fresh, infer_csv_end_timestamp, write_sync_metadata
from data.timeout import run_with_timeout
from tqdm import tqdm

DEFAULT_SYNC_MAX_ATTEMPTS = 3
DEFAULT_SYNC_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_REMOTE_SUBTASK_TIMEOUT_SECONDS = 60.0
DEFAULT_INCREMENTAL_LOOKBACK_DAYS = 5


@dataclass(frozen=True)
class SymbolSyncResult:
    symbol: str
    path: str | None
    skipped: bool
    attempts: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None


@dataclass(frozen=True)
class BatchSyncSummary:
    output_dir: str
    elapsed_seconds: float
    results: list[SymbolSyncResult]
    summary_path: str | None = None
    manifest_path: str | None = None
    failures_path: str | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for result in self.results if result.succeeded and not result.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.skipped)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded - self.skipped

    @property
    def retried_symbols(self) -> int:
        return sum(1 for result in self.results if result.attempts > 1)

    @property
    def extra_attempts(self) -> int:
        return sum(max(0, result.attempts - 1) for result in self.results)

    @property
    def successful_paths(self) -> list[str]:
        return [result.path for result in self.results if result.path is not None]

    def format_overview(self) -> str:
        return (
            "Sync summary: "
            f"{self.succeeded}/{self.total} succeeded, "
            f"{self.skipped} skipped, "
            f"{self.failed} failed, "
            f"{self.retried_symbols} retried symbols, "
            f"{self.extra_attempts} extra attempts, "
            f"elapsed {self.elapsed_seconds:.1f}s, "
            f"output {self.output_dir}"
        )

    def format_failure_summary(self, max_examples: int = 10) -> str | None:
        failed_results = [result for result in self.results if not result.succeeded]
        if not failed_results:
            return None

        examples = ", ".join(
            f"{result.symbol} ({result.attempts} attempts, {result.error})"
            for result in failed_results[:max_examples]
        )
        return f"Failed to sync {len(failed_results)} symbol(s). Examples: {examples}"


class DownloadRetryExhaustedError(RuntimeError):
    def __init__(self, *, symbol: str, attempts: int, last_error: Exception) -> None:
        self.symbol = symbol
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{symbol} failed after {attempts} attempts: {type(last_error).__name__}: {last_error}"
        )


def _download_daily_frame(
    provider: AKShareAshareProvider,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
):
    return provider.download_daily_frame(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )


def _download_daily_with_retries(
    *,
    symbol: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    effective_start_date: str,
) -> dict[str, str | int]:
    provider = AKShareAshareProvider()
    last_error: Exception | None = None
    target = _resolve_daily_target_path(output_dir=output_dir, symbol=symbol)

    for attempt in range(1, max_attempts + 1):
        try:
            timed = run_with_timeout(
                lambda: _download_daily_frame(
                    provider=provider,
                    symbol=symbol,
                    start_date=effective_start_date,
                    end_date=end_date,
                    adjust=adjust,
                ),
                timeout_seconds=timeout_seconds,
                task_label=f"daily-sync:{symbol}",
            )
            merged = _merge_incremental_daily_frame(
                target=target,
                downloaded=timed.value,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(target, index=False)
            return {"symbol": symbol, "path": str(target), "attempts": attempt}
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts and retry_backoff_seconds > 0:
                sleep(retry_backoff_seconds * attempt)

    assert last_error is not None
    raise DownloadRetryExhaustedError(
        symbol=symbol,
        attempts=max_attempts,
        last_error=last_error,
    ) from last_error


def _resolve_daily_target_path(*, output_dir: Path, symbol: str) -> Path:
    return output_dir / f"{symbol}.csv"


def _resolve_incremental_start_date(
    *,
    target: Path,
    requested_start_date: str,
    requested_end_date: str,
    incremental_lookback_days: int,
) -> str | None:
    latest = infer_csv_end_timestamp(target, date_columns=["timestamp", "date", "trade_date"])
    if latest is None:
        return requested_start_date

    requested_end = pd.Timestamp(requested_end_date)
    if latest >= requested_end:
        return None

    overlap_start = latest - pd.Timedelta(days=max(0, int(incremental_lookback_days)))
    return max(pd.Timestamp(requested_start_date), overlap_start).strftime("%Y%m%d")


def _merge_incremental_daily_frame(*, target: Path, downloaded: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if target.exists():
        existing = pd.read_csv(target)
        if not existing.empty:
            frames.append(existing)
    frames.append(downloaded)

    if not frames:
        return downloaded

    merged = pd.concat(frames, ignore_index=True)
    if "timestamp" in merged.columns:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
        merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def _sync_daily_symbol(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    incremental_lookback_days: int,
    overwrite: bool,
) -> SymbolSyncResult:
    target = _resolve_daily_target_path(output_dir=output_dir, symbol=symbol)
    if not overwrite and csv_target_is_fresh(
        target,
        end_date=end_date,
        date_columns=["timestamp", "date", "trade_date"],
        metadata_hint="timestamp",
    ):
        return SymbolSyncResult(
            symbol=symbol,
            path=str(target),
            skipped=True,
            attempts=0,
        )
    effective_start_date = start_date
    if not overwrite:
        incremental_start_date = _resolve_incremental_start_date(
            target=target,
            requested_start_date=start_date,
            requested_end_date=end_date,
            incremental_lookback_days=incremental_lookback_days,
        )
        if incremental_start_date is None:
            return SymbolSyncResult(
                symbol=symbol,
                path=str(target),
                skipped=True,
                attempts=0,
            )
        effective_start_date = incremental_start_date

    result = _download_daily_with_retries(
        symbol=symbol,
        end_date=end_date,
        adjust=adjust,
        output_dir=output_dir,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        effective_start_date=effective_start_date,
    )
    write_sync_metadata(target, synced_end_date=end_date, extra={"date_column": "timestamp"})
    return SymbolSyncResult(
        symbol=symbol,
        path=str(result["path"]),
        skipped=False,
        attempts=int(result["attempts"]),
    )


def _print_sync_summary(summary: BatchSyncSummary) -> None:
    print(summary.format_overview())
    failure_summary = summary.format_failure_summary()
    if failure_summary:
        print(failure_summary)


def sync_ashare_daily(
    symbols: list[str],
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: str,
    max_workers: int | None = None,
    max_attempts: int = DEFAULT_SYNC_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_SYNC_RETRY_BACKOFF_SECONDS,
    timeout_seconds: float | None = DEFAULT_REMOTE_SUBTASK_TIMEOUT_SECONDS,
    incremental_lookback_days: int = DEFAULT_INCREMENTAL_LOOKBACK_DAYS,
    show_progress: bool = True,
    print_summary: bool = True,
    allow_partial: bool = True,
    overwrite: bool = False,
) -> BatchSyncSummary:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tracker = PipelineRunTracker(
        pipeline_name="daily_sync",
        output_dir=target,
        item_label="symbol",
        items=[str(symbol) for symbol in symbols],
        options={
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjust": str(adjust),
            "max_workers": int(max_workers) if max_workers is not None else None,
            "max_attempts": int(max_attempts),
            "retry_backoff_seconds": float(retry_backoff_seconds),
            "timeout_seconds": float(timeout_seconds) if timeout_seconds is not None else None,
            "incremental_lookback_days": int(incremental_lookback_days),
            "allow_partial": bool(allow_partial),
            "overwrite": bool(overwrite),
        },
    )
    results_by_symbol: dict[str, SymbolSyncResult] = {}
    started_at = monotonic()
    execution = run_adaptive_io_tasks(
        items=symbols,
        worker_fn=lambda symbol: _sync_daily_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            output_dir=target,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            incremental_lookback_days=incremental_lookback_days,
            overwrite=overwrite,
        ),
        progress_desc="Syncing daily bars",
        progress_unit="symbol",
        show_progress=show_progress,
        max_workers=max_workers,
        progress_factory=tqdm,
    )
    for outcome in execution.outcomes:
        symbol = str(outcome.item)
        if outcome.succeeded:
            assert outcome.value is not None
            results_by_symbol[symbol] = outcome.value
            continue

        exc = outcome.error
        assert exc is not None
        attempts = int(getattr(exc, "attempts", max_attempts))
        error = getattr(exc, "last_error", exc)
        results_by_symbol[symbol] = SymbolSyncResult(
            symbol=symbol,
            path=None,
            skipped=False,
            attempts=attempts,
            error=f"{type(error).__name__}: {error}",
        )

    for symbol in symbols:
        result = results_by_symbol[str(symbol)]
        if result.succeeded:
            tracker.record_result(
                str(symbol),
                {
                    "symbol": result.symbol,
                    "path": result.path,
                    "skipped": result.skipped,
                    "attempts": result.attempts,
                },
            )
        else:
            tracker.record_failure(
                item=str(symbol),
                error=str(result.error or "unknown error"),
                attempts=result.attempts,
            )

    tracker_status = "partial_success" if any(not result.succeeded for result in results_by_symbol.values()) else "completed"
    tracker.finalize(status=tracker_status)
    summary = BatchSyncSummary(
        output_dir=str(target),
        elapsed_seconds=monotonic() - started_at,
        results=[results_by_symbol[symbol] for symbol in symbols],
        summary_path=str(tracker.summary_path),
        manifest_path=str(tracker.manifest_path),
        failures_path=str(tracker.failures_path),
    )

    if print_summary:
        _print_sync_summary(summary)

    if summary.failed and not allow_partial:
        raise RuntimeError(summary.format_failure_summary() or "Daily data sync completed with failures.")
    if not summary.successful_paths:
        raise RuntimeError("Daily data sync failed for all requested symbols.")
    return summary


def resolve_sync_symbols(
    *,
    symbols: list[str] | None,
    universe: str | None,
    universe_root: str,
    adjust: str,
) -> list[str]:
    loader = CsvDataProvider(universe_root=universe_root, default_adjust=adjust)
    if symbols:
        return [loader.normalize_symbol(symbol) for symbol in symbols]
    if universe:
        resolved = loader.load_universe(market="ashare", universe=universe, date=None)
        if not resolved:
            raise ValueError(f"Universe produced no symbols: {universe}")
        return resolved
    raise ValueError("Provide either symbols or a universe for daily data sync.")
