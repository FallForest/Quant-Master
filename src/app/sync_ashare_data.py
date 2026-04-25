from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

from data.providers.akshare_provider import AKShareAshareProvider
from data.providers.csv_provider import CsvDataProvider
from data.io_parallel import run_adaptive_io_tasks
from tqdm import tqdm

DEFAULT_SYNC_MAX_ATTEMPTS = 3
DEFAULT_SYNC_RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class SymbolSyncResult:
    symbol: str
    path: str | None
    attempts: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class BatchSyncSummary:
    output_dir: str
    elapsed_seconds: float
    results: list[SymbolSyncResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for result in self.results if result.succeeded)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

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
        return (
            f"Skipped {len(failed_results)} symbols during sync due to download errors. "
            f"Examples: {examples}"
        )


class DownloadRetryExhaustedError(RuntimeError):
    def __init__(self, *, symbol: str, attempts: int, last_error: Exception) -> None:
        self.symbol = symbol
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{symbol} failed after {attempts} attempts: {type(last_error).__name__}: {last_error}"
        )


def _download_daily_to_csv(
    provider: AKShareAshareProvider,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
) -> str:
    normalized = provider.download_daily_to_csv(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
        output_path=output_dir / f"{symbol}.csv",
    )
    return str(normalized)


def _download_daily_with_retries(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> dict[str, str | int]:
    provider = AKShareAshareProvider()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            path = _download_daily_to_csv(
                provider=provider,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                output_dir=output_dir,
            )
            return {"symbol": symbol, "path": path, "attempts": attempt}
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
    show_progress: bool = True,
    print_summary: bool = True,
    allow_partial: bool = False,
) -> BatchSyncSummary:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    completed_paths: dict[str, str] = {}
    attempts_by_symbol: dict[str, int] = {}
    started_at = monotonic()
    execution = run_adaptive_io_tasks(
        items=symbols,
        worker_fn=lambda symbol: _download_daily_with_retries(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            output_dir=target,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        ),
        progress_desc="Syncing daily bars",
        progress_unit="symbol",
        show_progress=show_progress,
        max_workers=max_workers,
        progress_factory=tqdm,
    )
    failure_errors: dict[str, str] = {}
    for outcome in execution.outcomes:
        symbol = str(outcome.item)
        if outcome.succeeded:
            assert outcome.value is not None
            completed_paths[symbol] = str(outcome.value["path"])
            attempts_by_symbol[symbol] = int(outcome.value["attempts"])
            continue

        exc = outcome.error
        assert exc is not None
        attempts = int(getattr(exc, "attempts", max_attempts))
        attempts_by_symbol[symbol] = attempts
        error = getattr(exc, "last_error", exc)
        failure_errors[symbol] = f"{type(error).__name__}: {error}"

    summary = BatchSyncSummary(
        output_dir=str(target),
        elapsed_seconds=monotonic() - started_at,
        results=[
            SymbolSyncResult(
                symbol=symbol,
                path=completed_paths.get(symbol),
                attempts=attempts_by_symbol.get(symbol, max_attempts),
                error=failure_errors.get(symbol),
            )
            for symbol in symbols
        ],
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
