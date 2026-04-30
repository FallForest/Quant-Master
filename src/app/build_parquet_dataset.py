from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data.providers.csv_provider import CsvDataProvider
from data.io_parallel import format_adaptive_io_failures, run_adaptive_io_tasks
from data.pipeline_tracking import PipelineRunTracker
from data.providers.parquet_provider import ParquetDataProvider


@dataclass(frozen=True)
class ParquetBuildResult:
    symbol: str
    path: str | None
    skipped: bool
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None


@dataclass(frozen=True)
class ParquetBuildSummary:
    output_root: str
    elapsed_seconds: float
    results: list[ParquetBuildResult]
    summary_path: str | None = None
    manifest_path: str | None = None
    failures_path: str | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def built(self) -> int:
        return sum(1 for result in self.results if result.succeeded and not result.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.skipped)

    @property
    def failed(self) -> int:
        return self.total - self.built - self.skipped

    @property
    def successful_paths(self) -> list[str]:
        return [result.path for result in self.results if result.path is not None]

    def format_overview(self) -> str:
        return (
            "Parquet build summary: "
            f"{self.built}/{self.total} built, "
            f"{self.skipped} skipped, "
            f"{self.failed} failed, "
            f"elapsed {self.elapsed_seconds:.1f}s, "
            f"output {self.output_root}"
        )

    def format_failure_summary(self, max_examples: int = 5) -> str | None:
        failed_results = [result for result in self.results if result.error]
        if not failed_results:
            return None
        examples = ", ".join(
            f"{result.symbol} ({result.error})"
            for result in failed_results[:max_examples]
        )
        return f"Failed to build parquet for {len(failed_results)} symbol(s). Examples: {examples}"


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
    overwrite: bool = False,
    allow_partial: bool = True,
    show_progress: bool = True,
    print_summary: bool = True,
) -> ParquetBuildSummary:
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
    target_root = Path(output_root) / market / target.normalize_timeframe_dir(timeframe) / target.normalize_adjust_dir(adjust)
    tracker = PipelineRunTracker(
        pipeline_name="parquet_build",
        output_dir=target_root,
        item_label="symbol",
        items=[str(symbol) for symbol in resolved_symbols],
        options={
            "market": str(market),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "timeframe": str(timeframe),
            "adjust": str(adjust),
            "max_workers": int(max_workers) if max_workers is not None else None,
            "allow_partial": bool(allow_partial),
            "overwrite": bool(overwrite),
        },
    )

    def _build_one_symbol(symbol: str) -> ParquetBuildResult:
        source_path = source.resolve_data_path(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            suffix=source.file_suffix,
        )
        target_path = target.resolve_data_path(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            suffix=target.file_suffix,
        )
        if (
            not overwrite
            and source_path.exists()
            and target_path.exists()
            and target_path.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return ParquetBuildResult(symbol=symbol, path=str(target_path), skipped=True)

        bars = source.load_bars(
            symbol=symbol,
            market=market,
            start=datetime(1900, 1, 1),
            end=datetime(2100, 1, 1),
            timeframe=timeframe,
        )
        if bars.empty:
            return ParquetBuildResult(
                symbol=symbol,
                path=None,
                skipped=False,
                error="No source bars found",
            )

        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            bars.to_parquet(target_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Building parquet datasets requires `pyarrow`. Run `py -m pip install -r requirements.txt` first."
            ) from exc
        return ParquetBuildResult(symbol=symbol, path=str(target_path), skipped=False)

    started_at = datetime.now()
    execution = run_adaptive_io_tasks(
        items=resolved_symbols,
        worker_fn=_build_one_symbol,
        progress_desc="Building parquet dataset",
        progress_unit="symbol",
        max_workers=max_workers,
        show_progress=show_progress,
    )
    results: list[ParquetBuildResult] = []
    for outcome in execution.outcomes:
        symbol = str(outcome.item)
        if outcome.succeeded:
            assert outcome.value is not None
            result = outcome.value
            results.append(result)
            if result.error:
                tracker.record_failure(item=symbol, error=str(result.error))
            else:
                tracker.record_result(
                    symbol,
                    {
                        "symbol": result.symbol,
                        "path": result.path,
                        "skipped": result.skipped,
                    },
                )
            continue
        assert outcome.error is not None
        error = f"{type(outcome.error).__name__}: {outcome.error}"
        results.append(
            ParquetBuildResult(
                symbol=symbol,
                path=None,
                skipped=False,
                error=error,
            )
        )
        tracker.record_failure(item=symbol, error=error)
    tracker_status = "partial_success" if any(result.error for result in results) else "completed"
    tracker.finalize(status=tracker_status)
    summary = ParquetBuildSummary(
        output_root=str(Path(output_root)),
        elapsed_seconds=(datetime.now() - started_at).total_seconds(),
        results=results,
        summary_path=str(tracker.summary_path),
        manifest_path=str(tracker.manifest_path),
        failures_path=str(tracker.failures_path),
    )
    if print_summary:
        print(summary.format_overview())
        failure_summary = summary.format_failure_summary()
        if failure_summary:
            print(failure_summary)
    if summary.failed and not allow_partial:
        failure_summary = summary.format_failure_summary()
        if failure_summary:
            raise RuntimeError(failure_summary)
        failures = [outcome for outcome in execution.outcomes if not outcome.succeeded]
        examples = format_adaptive_io_failures(failures)
        raise RuntimeError(f"Failed to build parquet files for {len(failures)} symbol(s). Examples: {examples}")
    if not summary.successful_paths:
        raise RuntimeError("Parquet build produced no output files.")
    return summary
