from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep
from threading import Lock
import numpy as np
import pandas as pd

from data.io_parallel import AdaptiveIoOutcome, resolve_auto_io_workers, run_adaptive_io_tasks
from data.pipeline_tracking import PipelineRunTracker
from data.sync_metadata import csv_target_is_fresh, load_sync_metadata, write_sync_metadata
from data.timeout import run_with_timeout


DEFAULT_REFERENCE_SUBTASK_TIMEOUT_SECONDS = 60.0
_CNINFO_ENCKEY_LOCK = Lock()
_CNINFO_ENCKEY_CONTEXT = None


REFERENCE_SYNC_SCOPES: dict[str, tuple[str, ...]] = {
    "all": ("benchmark", "fundamentals", "industry", "dividends"),
    "benchmark-only": ("benchmark",),
    "fundamentals-only": ("fundamentals",),
    "industry-only": ("industry",),
    "dividends-only": ("dividends",),
}


@dataclass(frozen=True)
class ReferenceSymbolSyncResult:
    symbol: str
    path: str | None
    industry_path: str | None
    dividend_path: str | None
    skipped: bool
    attempts: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (self.path is not None or self.industry_path is not None or self.dividend_path is not None) and self.error is None


@dataclass(frozen=True)
class ReferenceSyncTask:
    task_id: str
    scope: str
    symbol: str | None = None


@dataclass(frozen=True)
class ReferenceSyncSummary:
    benchmark_path: str
    elapsed_seconds: float
    results: list[ReferenceSymbolSyncResult]
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
    def fundamental_paths(self) -> list[str]:
        return [result.path for result in self.results if result.path is not None]

    @property
    def industry_paths(self) -> list[str]:
        return [result.industry_path for result in self.results if result.industry_path is not None]

    @property
    def dividend_paths(self) -> list[str]:
        return [result.dividend_path for result in self.results if result.dividend_path is not None]

    def format_overview(self) -> str:
        return (
            "Reference sync summary: "
            f"{self.succeeded}/{self.total} downloaded, "
            f"{self.skipped} skipped, "
            f"{self.failed} failed, "
            f"elapsed {self.elapsed_seconds:.1f}s, "
            f"benchmark {self.benchmark_path}"
        )

    def format_failure_summary(self, max_examples: int = 10) -> str | None:
        failed_results = [result for result in self.results if result.error]
        if not failed_results:
            return None
        examples = ", ".join(
            f"{result.symbol} ({result.attempts} attempts, {result.error})"
            for result in failed_results[:max_examples]
        )
        return f"Reference sync failed for {len(failed_results)} symbol(s). Examples: {examples}"


def sync_ashare_reference_data(
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    benchmark_symbol: str = "sh000300",
    reference_root: str = "data/reference",
    max_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
    timeout_seconds: float | None = DEFAULT_REFERENCE_SUBTASK_TIMEOUT_SECONDS,
    max_workers: int | None = None,
    bundle_workers: int | None = None,
    show_progress: bool = True,
    print_summary: bool = True,
    overwrite: bool = False,
    allow_partial: bool = True,
    scope: str = "all",
) -> ReferenceSyncSummary:
    active_scopes = _resolve_reference_scopes(scope)
    try:
        ak = import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError("AKShare is not installed. Run `py -m pip install -r requirements.txt` first.") from exc

    started_at = monotonic()
    tracker_items = _build_reference_tracker_items(
        symbols=symbols,
        benchmark_symbol=benchmark_symbol,
        active_scopes=active_scopes,
    )
    effective_max_workers = _resolve_reference_task_workers(
        task_count=len(tracker_items),
        max_workers=max_workers,
        bundle_workers=bundle_workers,
        active_scope_count=sum(1 for scope_name in ("fundamentals", "industry", "dividends") if scope_name in active_scopes),
    )
    tracker = PipelineRunTracker(
        pipeline_name="reference_sync",
        output_dir=Path(reference_root) / "ashare",
        item_label="task",
        items=tracker_items,
        options={
            "start_date": str(start_date),
            "end_date": str(end_date),
            "benchmark_symbol": str(benchmark_symbol),
            "scope": str(scope),
            "max_attempts": int(max_attempts),
            "retry_backoff_seconds": float(retry_backoff_seconds),
            "timeout_seconds": float(timeout_seconds) if timeout_seconds is not None else None,
            "max_workers": int(max_workers) if max_workers is not None else None,
            "bundle_workers": int(bundle_workers) if bundle_workers is not None else None,
            "effective_max_workers": int(effective_max_workers),
            "allow_partial": bool(allow_partial),
            "overwrite": bool(overwrite),
        },
    )
    benchmark_path = _resolve_benchmark_target_path(
        benchmark_symbol=benchmark_symbol,
        reference_root=reference_root,
    )
    symbol_states: dict[str, dict[str, object]] = {
        str(symbol): {
            "path": None,
            "industry_path": None,
            "dividend_path": None,
            "attempts": [],
            "errors": [],
            "completed_scopes": 0,
            "skipped_scopes": 0,
        }
        for symbol in symbols
    }
    if "benchmark" in active_scopes:
        benchmark_task = f"benchmark:{benchmark_symbol}"
        try:
            benchmark_path, benchmark_skipped = _sync_benchmark_frame(
                ak=ak,
                benchmark_symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                reference_root=reference_root,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
            )
            tracker.record_result(
                benchmark_task,
                {
                    "task": benchmark_task,
                    "scope": "benchmark",
                    "symbol": benchmark_symbol,
                    "path": benchmark_path,
                    "skipped": benchmark_skipped,
                    "attempts": 1,
                },
            )
        except Exception as exc:  # noqa: BLE001
            tracker.record_failure(item=benchmark_task, error=f"{type(exc).__name__}: {exc}", attempts=max_attempts)
            tracker.finalize(status="failed")
            raise

    skipped_results, pending_tasks = _resolve_reference_sync_plan(
        symbols=symbols,
        end_date=end_date,
        reference_root=reference_root,
        active_scopes=active_scopes,
        overwrite=overwrite,
    )
    if "dividends" in active_scopes and pending_tasks:
        _get_cninfo_accept_enckey()
    for result in skipped_results:
        _merge_reference_symbol_result(symbol_states, result, active_scopes=active_scopes)
        for task_payload in _build_reference_result_task_payloads(result, active_scopes=active_scopes):
            tracker.record_result(str(task_payload["task"]), task_payload)

    run_adaptive_io_tasks(
        items=pending_tasks,
        worker_fn=lambda task: _run_reference_sync_task(
            ak=ak,
            task=task,
            end_date=end_date,
            reference_root=reference_root,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        ),
        progress_desc="Syncing reference data",
        progress_unit="task",
        show_progress=show_progress,
        max_workers=effective_max_workers,
        on_outcome=lambda outcome: _record_reference_sync_task_outcome(
            tracker=tracker,
            symbol_states=symbol_states,
            outcome=outcome,
            max_attempts=max_attempts,
        ),
    )
    ordered_results = [
        _build_reference_symbol_result(
            symbol=str(symbol),
            state=symbol_states[str(symbol)],
            active_scopes=active_scopes,
        )
        for symbol in symbols
    ]
    tracker_status = "partial_success" if any(result.error for result in ordered_results) else "completed"
    tracker.finalize(status=tracker_status)
    summary = ReferenceSyncSummary(
        benchmark_path=benchmark_path,
        elapsed_seconds=monotonic() - started_at,
        results=ordered_results,
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
        raise RuntimeError(summary.format_failure_summary() or "Reference data sync completed with failures.")
    if "fundamentals" in active_scopes and not summary.fundamental_paths and symbols:
        raise RuntimeError("Reference data sync failed for all requested symbols.")
    if "industry" in active_scopes and not summary.industry_paths and symbols:
        raise RuntimeError("Industry reference sync failed for all requested symbols.")
    if "dividends" in active_scopes and not summary.dividend_paths and symbols:
        raise RuntimeError("Dividend reference sync failed for all requested symbols.")
    return summary


def resolve_symbols_needing_reference_sync(
    *,
    symbols: list[str],
    end_date: str,
    reference_root: str = "data/reference",
    scope: str = "all",
    overwrite: bool = False,
) -> tuple[list[ReferenceSymbolSyncResult], list[str]]:
    active_scopes = _resolve_reference_scopes(scope)
    return _resolve_symbols_needing_reference_sync(
        symbols=symbols,
        end_date=end_date,
        reference_root=reference_root,
        active_scopes=active_scopes,
        overwrite=overwrite,
    )


def _resolve_symbols_needing_reference_sync(
    *,
    symbols: list[str],
    end_date: str,
    reference_root: str,
    active_scopes: tuple[str, ...],
    overwrite: bool,
) -> tuple[list[ReferenceSymbolSyncResult], list[str]]:
    skipped_results: list[ReferenceSymbolSyncResult] = []
    pending_symbols: list[str] = []
    if overwrite:
        return skipped_results, [str(symbol) for symbol in symbols]
    for symbol in symbols:
        targets = _resolve_reference_symbol_targets(symbol=str(symbol), reference_root=reference_root)
        paths: dict[str, str | None] = {
            "fundamentals": None,
            "industry": None,
            "dividends": None,
        }
        scope_freshness: list[bool] = []
        if "fundamentals" in active_scopes:
            fresh = _reference_sync_is_fresh(targets["fundamentals"], end_date=end_date)
            scope_freshness.append(fresh)
            if fresh:
                paths["fundamentals"] = str(targets["fundamentals"])
        if "industry" in active_scopes:
            fresh = _reference_sync_is_fresh(targets["industry"], end_date=end_date, date_column="change_date")
            scope_freshness.append(fresh)
            if fresh:
                paths["industry"] = str(targets["industry"])
        if "dividends" in active_scopes:
            fresh = _reference_sync_is_fresh(targets["dividends"], end_date=end_date, date_column="event_date")
            scope_freshness.append(fresh)
            if fresh:
                paths["dividends"] = str(targets["dividends"])
        if scope_freshness and all(scope_freshness):
            skipped_results.append(
                ReferenceSymbolSyncResult(
                    symbol=str(symbol),
                    path=paths["fundamentals"],
                    industry_path=paths["industry"],
                    dividend_path=paths["dividends"],
                    skipped=True,
                    attempts=0,
                )
            )
            continue
        pending_symbols.append(str(symbol))
    return skipped_results, pending_symbols


def _resolve_reference_sync_plan(
    *,
    symbols: list[str],
    end_date: str,
    reference_root: str,
    active_scopes: tuple[str, ...],
    overwrite: bool,
) -> tuple[list[ReferenceSymbolSyncResult], list[ReferenceSyncTask]]:
    skipped_results, pending_symbols = _resolve_symbols_needing_reference_sync(
        symbols=symbols,
        end_date=end_date,
        reference_root=reference_root,
        active_scopes=active_scopes,
        overwrite=overwrite,
    )
    pending_tasks: list[ReferenceSyncTask] = []
    for symbol in pending_symbols:
        targets = _resolve_reference_symbol_targets(symbol=str(symbol), reference_root=reference_root)
        if "fundamentals" in active_scopes and (
            overwrite or not _reference_sync_is_fresh(targets["fundamentals"], end_date=end_date)
        ):
            pending_tasks.append(ReferenceSyncTask(task_id=f"fundamentals:{symbol}", scope="fundamentals", symbol=str(symbol)))
        if "industry" in active_scopes and (
            overwrite or not _reference_sync_is_fresh(targets["industry"], end_date=end_date, date_column="change_date")
        ):
            pending_tasks.append(ReferenceSyncTask(task_id=f"industry:{symbol}", scope="industry", symbol=str(symbol)))
        if "dividends" in active_scopes and (
            overwrite or not _reference_sync_is_fresh(targets["dividends"], end_date=end_date, date_column="event_date")
        ):
            pending_tasks.append(ReferenceSyncTask(task_id=f"dividends:{symbol}", scope="dividends", symbol=str(symbol)))
    return skipped_results, pending_tasks


def _build_reference_tracker_items(
    *,
    symbols: list[str],
    benchmark_symbol: str,
    active_scopes: tuple[str, ...],
) -> list[str]:
    items: list[str] = []
    if "benchmark" in active_scopes:
        items.append(f"benchmark:{benchmark_symbol}")
    for symbol in symbols:
        if "fundamentals" in active_scopes:
            items.append(f"fundamentals:{symbol}")
        if "industry" in active_scopes:
            items.append(f"industry:{symbol}")
        if "dividends" in active_scopes:
            items.append(f"dividends:{symbol}")
    return items


def _resolve_reference_task_workers(
    *,
    task_count: int,
    max_workers: int | None,
    bundle_workers: int | None,
    active_scope_count: int,
) -> int:
    if task_count <= 0:
        return 1
    outer_workers = resolve_auto_io_workers(max_workers)
    if bundle_workers is not None:
        bundle_multiplier = max(1, int(bundle_workers))
    else:
        bundle_multiplier = min(2, max(1, int(active_scope_count)))
    return min(task_count, outer_workers * bundle_multiplier)


def _build_reference_result_task_payloads(
    result: ReferenceSymbolSyncResult,
    *,
    active_scopes: tuple[str, ...],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    if "fundamentals" in active_scopes and result.path is not None:
        payloads.append(
            {
                "task": f"fundamentals:{result.symbol}",
                "scope": "fundamentals",
                "symbol": result.symbol,
                "path": result.path,
                "skipped": result.skipped,
                "attempts": result.attempts,
            }
        )
    if "industry" in active_scopes and result.industry_path is not None:
        payloads.append(
            {
                "task": f"industry:{result.symbol}",
                "scope": "industry",
                "symbol": result.symbol,
                "path": result.industry_path,
                "skipped": result.skipped,
                "attempts": result.attempts,
            }
        )
    if "dividends" in active_scopes and result.dividend_path is not None:
        payloads.append(
            {
                "task": f"dividends:{result.symbol}",
                "scope": "dividends",
                "symbol": result.symbol,
                "path": result.dividend_path,
                "skipped": result.skipped,
                "attempts": result.attempts,
            }
        )
    return payloads


def _merge_reference_symbol_result(
    symbol_states: dict[str, dict[str, object]],
    result: ReferenceSymbolSyncResult,
    *,
    active_scopes: tuple[str, ...],
) -> None:
    state = symbol_states[result.symbol]
    if result.path is not None and "fundamentals" in active_scopes:
        state["path"] = result.path
        state["completed_scopes"] = int(state["completed_scopes"]) + 1
        if result.skipped:
            state["skipped_scopes"] = int(state["skipped_scopes"]) + 1
    if result.industry_path is not None and "industry" in active_scopes:
        state["industry_path"] = result.industry_path
        state["completed_scopes"] = int(state["completed_scopes"]) + 1
        if result.skipped:
            state["skipped_scopes"] = int(state["skipped_scopes"]) + 1
    if result.dividend_path is not None and "dividends" in active_scopes:
        state["dividend_path"] = result.dividend_path
        state["completed_scopes"] = int(state["completed_scopes"]) + 1
        if result.skipped:
            state["skipped_scopes"] = int(state["skipped_scopes"]) + 1
    cast_attempts = state["attempts"]
    assert isinstance(cast_attempts, list)
    cast_attempts.append(int(result.attempts))
    if result.error:
        cast_errors = state["errors"]
        assert isinstance(cast_errors, list)
        cast_errors.append(str(result.error))


def _record_reference_sync_task_outcome(
    *,
    tracker: PipelineRunTracker,
    symbol_states: dict[str, dict[str, object]],
    outcome: AdaptiveIoOutcome[ReferenceSyncTask, object],
    max_attempts: int,
) -> None:
    task = outcome.item
    if outcome.succeeded:
        payload = _normalize_reference_task_payload(task=task, value=outcome.value)
        tracker.record_result(task.task_id, payload)
        symbol = str(payload["symbol"])
        state = symbol_states[symbol]
        scope = str(payload["scope"])
        if scope == "fundamentals":
            state["path"] = payload["path"]
        elif scope == "industry":
            state["industry_path"] = payload["path"]
        elif scope == "dividends":
            state["dividend_path"] = payload["path"]
        state["completed_scopes"] = int(state["completed_scopes"]) + 1
        if bool(payload.get("skipped")):
            state["skipped_scopes"] = int(state["skipped_scopes"]) + 1
        cast_attempts = state["attempts"]
        assert isinstance(cast_attempts, list)
        cast_attempts.append(int(payload["attempts"]))
        return

    exc = outcome.error
    assert exc is not None
    attempts = int(getattr(exc, "attempts", max_attempts))
    tracker.record_failure(item=task.task_id, error=f"{type(exc).__name__}: {exc}", attempts=attempts)
    assert task.symbol is not None
    state = symbol_states[task.symbol]
    cast_attempts = state["attempts"]
    cast_errors = state["errors"]
    assert isinstance(cast_attempts, list)
    assert isinstance(cast_errors, list)
    cast_attempts.append(attempts)
    cast_errors.append(f"{task.scope}={type(exc).__name__}: {exc}")


def _normalize_reference_task_payload(*, task: ReferenceSyncTask, value: object) -> dict[str, object]:
    assert task.symbol is not None
    if task.scope == "fundamentals":
        assert isinstance(value, ReferenceSymbolSyncResult)
        return {
            "task": task.task_id,
            "scope": task.scope,
            "symbol": task.symbol,
            "path": value.path,
            "skipped": value.skipped,
            "attempts": value.attempts,
        }
    assert isinstance(value, tuple)
    return {
        "task": task.task_id,
        "scope": task.scope,
        "symbol": task.symbol,
        "path": str(value[0]),
        "skipped": bool(value[1]),
        "attempts": 0 if bool(value[1]) else 1,
    }


def _build_reference_symbol_result(
    *,
    symbol: str,
    state: dict[str, object],
    active_scopes: tuple[str, ...],
) -> ReferenceSymbolSyncResult:
    attempts = state["attempts"]
    errors = state["errors"]
    assert isinstance(attempts, list)
    assert isinstance(errors, list)
    active_scope_count = sum(1 for scope_name in ("fundamentals", "industry", "dividends") if scope_name in active_scopes)
    return ReferenceSymbolSyncResult(
        symbol=symbol,
        path=state["path"] if isinstance(state["path"], str) else None,
        industry_path=state["industry_path"] if isinstance(state["industry_path"], str) else None,
        dividend_path=state["dividend_path"] if isinstance(state["dividend_path"], str) else None,
        skipped=active_scope_count > 0
        and int(state["completed_scopes"]) == active_scope_count
        and int(state["skipped_scopes"]) == active_scope_count
        and not errors,
        attempts=max(attempts) if attempts else 0,
        error="; ".join(errors) or None,
    )


def _resolve_reference_scopes(scope: str) -> tuple[str, ...]:
    normalized = str(scope).strip().lower()
    if normalized not in REFERENCE_SYNC_SCOPES:
        supported = ", ".join(sorted(REFERENCE_SYNC_SCOPES))
        raise ValueError(f"Unsupported reference sync scope: {scope}. Supported scopes: {supported}")
    return REFERENCE_SYNC_SCOPES[normalized]


def _resolve_reference_symbol_targets(*, symbol: str, reference_root: str) -> dict[str, Path]:
    root = Path(reference_root) / "ashare"
    return {
        "fundamentals": root / "fundamentals" / f"{symbol}.csv",
        "industry": root / "industry" / f"{symbol}.csv",
        "dividends": root / "dividends" / f"{symbol}.csv",
    }


def build_standardized_fundamental_frame(
    *,
    balance_frame: pd.DataFrame,
    profit_frame: pd.DataFrame,
    cash_flow_frame: pd.DataFrame,
) -> pd.DataFrame:
    balance = _select_and_rename_columns(
        balance_frame,
        mapping={
            "REPORT_DATE": "report_date",
            "NOTICE_DATE": "notice_date",
            "TOTAL_ASSETS": "total_assets",
            "TOTAL_LIABILITIES": "total_liabilities",
            "TOTAL_PARENT_EQUITY": "total_parent_equity",
            "TOTAL_EQUITY": "total_equity_fallback",
            "SHARE_CAPITAL": "share_capital",
            "INVENTORY": "inventory",
            "ACCOUNTS_RECE": "accounts_rece",
            "NOTE_RECE": "note_rece",
            "FIXED_ASSET": "fixed_asset",
            "INTANGIBLE_ASSET": "intangible_asset",
            "MONETARYFUNDS": "monetary_funds",
        },
    )
    profit = _select_and_rename_columns(
        profit_frame,
        mapping={
            "REPORT_DATE": "report_date",
            "NOTICE_DATE": "notice_date",
            "TOTAL_OPERATE_INCOME": "total_operate_income",
            "OPERATE_COST": "operate_cost",
            "OPERATE_PROFIT": "operate_profit",
            "PARENT_NETPROFIT": "parent_netprofit",
            "NETPROFIT": "netprofit_fallback",
        },
    )
    cash = _select_and_rename_columns(
        cash_flow_frame,
        mapping={
            "REPORT_DATE": "report_date",
            "NOTICE_DATE": "notice_date",
            "NETCASH_OPERATE": "netcash_operate",
            "CONSTRUCT_LONG_ASSET": "capex",
        },
    )

    merged = balance.merge(profit, on="report_date", how="outer", suffixes=("", "_profit"))
    merged = merged.merge(cash, on="report_date", how="outer", suffixes=("", "_cash"))
    merged = merged.sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last").reset_index(drop=True)

    merged["notice_date"] = _coalesce_datetime_columns(
        merged,
        columns=["notice_date", "notice_date_profit", "notice_date_cash"],
    )
    merged["available_date"] = merged["notice_date"].fillna(merged["report_date"])
    if "total_equity_fallback" in merged.columns:
        merged["total_parent_equity"] = merged["total_parent_equity"].fillna(merged["total_equity_fallback"])
    if "netprofit_fallback" in merged.columns:
        merged["parent_netprofit"] = merged["parent_netprofit"].fillna(merged["netprofit_fallback"])

    for column in [
        "total_assets",
        "total_liabilities",
        "total_parent_equity",
        "share_capital",
        "inventory",
        "accounts_rece",
        "note_rece",
        "fixed_asset",
        "intangible_asset",
        "monetary_funds",
        "total_operate_income",
        "operate_cost",
        "operate_profit",
        "parent_netprofit",
        "netcash_operate",
        "capex",
    ]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    for column in ["total_operate_income", "operate_cost", "operate_profit", "parent_netprofit", "netcash_operate", "capex"]:
        merged[f"{column}_single"] = _quarterize_cumulative_series(
            report_dates=merged["report_date"],
            values=merged[column],
        )
        merged[f"{column}_ttm"] = merged[f"{column}_single"].rolling(4, min_periods=4).sum()

    lag_assets = merged["total_assets"].shift(4)
    lag_inventory = merged["inventory"].shift(4)
    lag_receivables = merged["accounts_rece"].shift(4)
    lag_capex_ttm = merged["capex_ttm"].shift(4)
    merged["asset_growth"] = merged["total_assets"] / lag_assets.replace(0.0, np.nan) - 1.0
    merged["investment_to_assets"] = merged["capex_ttm"] / lag_assets.replace(0.0, np.nan)
    merged["accruals"] = (merged["parent_netprofit_ttm"] - merged["netcash_operate_ttm"]) / lag_assets.replace(
        0.0, np.nan
    )
    merged["inventory_growth"] = merged["inventory"] / lag_inventory.replace(0.0, np.nan) - 1.0
    merged["receivables_growth"] = merged["accounts_rece"] / lag_receivables.replace(0.0, np.nan) - 1.0
    merged["capex_growth"] = merged["capex_ttm"] / lag_capex_ttm.replace(0.0, np.nan) - 1.0

    return merged[
        [
            "report_date",
            "available_date",
            "total_assets",
            "total_liabilities",
            "total_parent_equity",
            "share_capital",
            "inventory",
            "accounts_rece",
            "note_rece",
            "fixed_asset",
            "intangible_asset",
            "monetary_funds",
            "total_operate_income_ttm",
            "operate_cost_ttm",
            "operate_profit_ttm",
            "parent_netprofit_ttm",
            "netcash_operate_ttm",
            "capex_ttm",
            "asset_growth",
            "investment_to_assets",
            "accruals",
            "inventory_growth",
            "receivables_growth",
            "capex_growth",
        ]
    ].sort_values("report_date").reset_index(drop=True)


def _sync_benchmark_frame(
    *,
    ak,
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    overwrite: bool,
) -> tuple[str, bool]:
    target = Path(reference_root) / "ashare" / "index" / f"{benchmark_symbol}.csv"
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date, date_column="timestamp"):
        return str(target), True

    raw = _call_with_retries(
        lambda: ak.stock_zh_index_daily_em(
            symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
        ),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"benchmark:{benchmark_symbol}",
    )
    normalized = raw.rename(columns={"date": "timestamp"}).copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target, index=False)
    write_sync_metadata(target, synced_end_date=end_date, extra={"date_column": "timestamp"})
    return str(target), False


def _sync_fundamental_frame(
    *,
    ak,
    symbol: str,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    overwrite: bool,
) -> ReferenceSymbolSyncResult:
    target = _resolve_reference_symbol_targets(symbol=symbol, reference_root=reference_root)["fundamentals"]
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date):
        return ReferenceSymbolSyncResult(
            symbol=str(symbol),
            path=str(target),
            industry_path=None,
            dividend_path=None,
            skipped=True,
            attempts=0,
        )

    em_symbol = _to_eastmoney_symbol(symbol)
    balance = _call_with_retries(
        lambda: ak.stock_balance_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"fundamentals-balance:{symbol}",
    )
    profit = _call_with_retries(
        lambda: ak.stock_profit_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"fundamentals-profit:{symbol}",
    )
    cash = _call_with_retries(
        lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"fundamentals-cash:{symbol}",
    )
    standardized = build_standardized_fundamental_frame(
        balance_frame=balance,
        profit_frame=profit,
        cash_flow_frame=cash,
    )
    standardized.insert(0, "symbol", str(symbol))
    target.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(target, index=False)
    write_sync_metadata(target, synced_end_date=end_date, extra={"date_column": "available_date"})
    return ReferenceSymbolSyncResult(
        symbol=str(symbol),
        path=str(target),
        industry_path=None,
        dividend_path=None,
        skipped=False,
        attempts=1,
    )


def _run_reference_sync_task(
    *,
    ak,
    task: ReferenceSyncTask,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    overwrite: bool,
) -> object:
    if task.scope == "fundamentals":
        assert task.symbol is not None
        return _sync_fundamental_frame(
            ak=ak,
            symbol=task.symbol,
            end_date=end_date,
            reference_root=reference_root,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        )
    if task.scope == "industry":
        assert task.symbol is not None
        return _sync_industry_frame(
            ak=ak,
            symbol=task.symbol,
            end_date=end_date,
            reference_root=reference_root,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        )
    if task.scope == "dividends":
        assert task.symbol is not None
        return _sync_dividend_frame(
            ak=ak,
            symbol=task.symbol,
            end_date=end_date,
            reference_root=reference_root,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        )
    raise ValueError(f"Unsupported reference sync task scope: {task.scope}")


def _sync_industry_frame(
    *,
    ak,
    symbol: str,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    overwrite: bool,
) -> tuple[str, bool]:
    target = _resolve_reference_symbol_targets(symbol=symbol, reference_root=reference_root)["industry"]
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date, date_column="change_date"):
        return str(target), True

    raw = _call_with_retries(
        lambda: ak.stock_industry_change_cninfo(symbol=str(symbol), start_date="19900101", end_date=end_date.replace("-", "")),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"industry:{symbol}",
    )
    standardized = _build_standardized_industry_frame(symbol=str(symbol), raw=raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(target, index=False)
    write_sync_metadata(target, synced_end_date=end_date, extra={"date_column": "change_date"})
    return str(target), False


def _sync_dividend_frame(
    *,
    ak,
    symbol: str,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    overwrite: bool,
) -> tuple[str, bool]:
    target = _resolve_reference_symbol_targets(symbol=symbol, reference_root=reference_root)["dividends"]
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date, date_column="event_date"):
        return str(target), True

    raw = _call_with_retries(
        lambda: _fetch_dividend_cninfo_frame(symbol=str(symbol)),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        timeout_label=f"dividends:{symbol}",
    )
    standardized = _build_standardized_dividend_frame(symbol=str(symbol), raw=raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(target, index=False)
    write_sync_metadata(target, synced_end_date=end_date, extra={"date_column": "event_date"})
    return str(target), False


def _fetch_dividend_cninfo_frame(*, symbol: str) -> pd.DataFrame:
    requests = import_module("requests")
    url = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"
    params = {"scode": str(symbol)}
    headers = {
        "Accept": "*/*",
        "Accept-Enckey": _get_cninfo_accept_enckey(),
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Length": "0",
        "Host": "webapi.cninfo.com.cn",
        "Origin": "http://webapi.cninfo.com.cn",
        "Pragma": "no-cache",
        "Proxy-Connection": "keep-alive",
        "Referer": "http://webapi.cninfo.com.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }
    response = requests.post(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data_json = response.json()
    records = data_json.get("records") or []
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "F006D": "实施方案公告日期",
            "F044V": "分红类型",
            "F011N": "转增比例",
            "F010N": "送股比例",
            "F012N": "派息比例",
            "F018D": "股权登记日",
            "F020D": "除权日",
            "F023D": "派息日",
            "F025D": "股份到账日",
            "F007V": "实施方案分红说明",
            "F001V": "报告时间",
        }
    )


def _get_cninfo_accept_enckey() -> str:
    global _CNINFO_ENCKEY_CONTEXT

    with _CNINFO_ENCKEY_LOCK:
        if _CNINFO_ENCKEY_CONTEXT is None:
            py_mini_racer = import_module("py_mini_racer")
            dividend_module = import_module("akshare.stock.stock_dividend_cninfo")
            get_file_content = getattr(dividend_module, "_get_file_content_ths")
            js_code = py_mini_racer.MiniRacer()
            js_code.eval(get_file_content("cninfo.js"))
            _CNINFO_ENCKEY_CONTEXT = js_code
        return str(_CNINFO_ENCKEY_CONTEXT.call("getResCode1"))


def _build_standardized_industry_frame(*, symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "change_date",
                "standard",
                "sector",
                "industry_level_1",
                "industry_level_2",
                "industry_level_3",
                "industry_code",
            ]
        )
    renamed = raw.rename(
        columns={
            "变更日期": "change_date",
            "分类标准": "standard",
            "行业门类": "sector",
            "行业大类": "industry_level_1",
            "行业中类": "industry_level_2",
            "行业次类": "industry_level_3",
            "行业编码": "industry_code",
            "证券代码": "symbol",
        }
    ).copy()
    renamed["symbol"] = str(symbol)
    renamed["change_date"] = pd.to_datetime(renamed["change_date"])
    columns = [
        "symbol",
        "change_date",
        "standard",
        "sector",
        "industry_level_1",
        "industry_level_2",
        "industry_level_3",
        "industry_code",
    ]
    for column in columns:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed[columns].sort_values(["change_date", "standard"]).reset_index(drop=True)


def _build_standardized_dividend_frame(*, symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "announcement_date",
                "record_date",
                "ex_date",
                "pay_date",
                "event_date",
                "cash_dividend_per_share",
                "dividend_type",
                "report_period",
            ]
        )

    renamed = raw.rename(
        columns={
            "实施方案公告日期": "announcement_date",
            "分红类型": "dividend_type",
            "派息比例": "cash_dividend_per_10_shares",
            "股权登记日": "record_date",
            "除权日": "ex_date",
            "派息日": "pay_date",
            "报告时间": "report_period",
        }
    ).copy()
    renamed["symbol"] = str(symbol)
    for column in ["announcement_date", "record_date", "ex_date", "pay_date"]:
        if column in renamed.columns:
            renamed[column] = pd.to_datetime(renamed[column], errors="coerce")
    renamed["cash_dividend_per_share"] = pd.to_numeric(
        renamed.get("cash_dividend_per_10_shares"),
        errors="coerce",
    ) / 10.0
    renamed["event_date"] = (
        renamed.get("ex_date")
        .fillna(renamed.get("pay_date"))
        .fillna(renamed.get("announcement_date"))
    )
    columns = [
        "symbol",
        "announcement_date",
        "record_date",
        "ex_date",
        "pay_date",
        "event_date",
        "cash_dividend_per_share",
        "dividend_type",
        "report_period",
    ]
    for column in columns:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed[columns].sort_values(["event_date", "announcement_date"]).reset_index(drop=True)


def _call_with_retries(
    fn,
    *,
    max_attempts: int,
    retry_backoff_seconds: float,
    timeout_seconds: float | None,
    timeout_label: str,
):
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return run_with_timeout(
                fn,
                timeout_seconds=timeout_seconds,
                task_label=timeout_label,
            ).value
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_attempts and retry_backoff_seconds > 0:
                sleep(retry_backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def _reference_sync_is_fresh(target: Path, *, end_date: str, date_column: str | None = None) -> bool:
    candidate_columns = [date_column] if date_column else ["timestamp", "date", "available_date", "report_date"]
    metadata_hint = None
    metadata = load_sync_metadata(target)
    if metadata is not None and metadata.get("date_column"):
        metadata_hint = str(metadata["date_column"])
        if metadata_hint not in candidate_columns:
            candidate_columns = [metadata_hint, *candidate_columns]
    return csv_target_is_fresh(
        target,
        end_date=end_date,
        date_columns=candidate_columns,
        metadata_hint=metadata_hint or date_column,
    )


def _resolve_benchmark_target_path(*, benchmark_symbol: str, reference_root: str) -> str:
    target = Path(reference_root) / "ashare" / "index" / f"{benchmark_symbol}.csv"
    return str(target) if target.exists() else ""


def _quarterize_cumulative_series(report_dates: pd.Series, values: pd.Series) -> pd.Series:
    quarterized = pd.Series(index=values.index, dtype=float)
    normalized_dates = pd.to_datetime(report_dates)
    normalized_values = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"report_date": normalized_dates, "value": normalized_values}).sort_values("report_date")

    for _, group in frame.groupby(frame["report_date"].dt.year):
        previous_value: float | None = None
        for index, row in group.iterrows():
            quarter = int(row["report_date"].quarter)
            value = row["value"]
            if pd.isna(value):
                quarterized.loc[index] = np.nan
            elif quarter == 1 or previous_value is None or pd.isna(previous_value):
                quarterized.loc[index] = float(value)
            else:
                quarterized.loc[index] = float(value) - float(previous_value)
            previous_value = value

    return quarterized.sort_index()


def _select_and_rename_columns(frame: pd.DataFrame, *, mapping: dict[str, str]) -> pd.DataFrame:
    existing = {source: target for source, target in mapping.items() if source in frame.columns}
    if "REPORT_DATE" not in existing:
        raise ValueError("Financial statement is missing REPORT_DATE.")
    selected = frame[list(existing.keys())].copy().rename(columns=existing)
    for target in mapping.values():
        if target not in selected.columns:
            selected[target] = pd.NA
    selected["report_date"] = pd.to_datetime(selected["report_date"])
    if "notice_date" in selected.columns:
        selected["notice_date"] = pd.to_datetime(selected["notice_date"])
    return selected


def _coalesce_datetime_columns(frame: pd.DataFrame, *, columns: list[str]) -> pd.Series:
    series = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for column in columns:
        if column not in frame.columns:
            continue
        series = series.fillna(pd.to_datetime(frame[column], errors="coerce"))
    return series


def _to_eastmoney_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    prefix = "SH" if text.startswith(("5", "6", "9")) else "SZ"
    return f"{prefix}{text}"
