from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pandas as pd

from data.io_parallel import run_adaptive_io_tasks


@dataclass(frozen=True)
class ReferenceSymbolSyncResult:
    symbol: str
    path: str | None
    skipped: bool
    attempts: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None


@dataclass(frozen=True)
class ReferenceSyncSummary:
    benchmark_path: str
    elapsed_seconds: float
    results: list[ReferenceSymbolSyncResult]

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
    max_workers: int | None = None,
    show_progress: bool = True,
    print_summary: bool = True,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> ReferenceSyncSummary:
    try:
        ak = import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError("AKShare is not installed. Run `py -m pip install -r requirements.txt` first.") from exc

    started_at = monotonic()
    benchmark_path = _sync_benchmark_frame(
        ak=ak,
        benchmark_symbol=benchmark_symbol,
        start_date=start_date,
        end_date=end_date,
        reference_root=reference_root,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        overwrite=overwrite,
    )

    results: dict[str, ReferenceSymbolSyncResult] = {}
    execution = run_adaptive_io_tasks(
        items=symbols,
        worker_fn=lambda symbol: _sync_fundamental_frame(
            ak=ak,
            symbol=symbol,
            end_date=end_date,
            reference_root=reference_root,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            overwrite=overwrite,
        ),
        progress_desc="Syncing reference data",
        progress_unit="symbol",
        show_progress=show_progress,
        max_workers=max_workers,
    )
    for outcome in execution.outcomes:
        symbol = str(outcome.item)
        if outcome.succeeded:
            assert outcome.value is not None
            results[symbol] = outcome.value
            continue

        exc = outcome.error
        assert exc is not None
        attempts = int(getattr(exc, "attempts", max_attempts))
        results[symbol] = ReferenceSymbolSyncResult(
            symbol=symbol,
            path=None,
            skipped=False,
            attempts=attempts,
            error=f"{type(exc).__name__}: {exc}",
        )

    ordered_results = [results[symbol] for symbol in symbols]
    summary = ReferenceSyncSummary(
        benchmark_path=benchmark_path,
        elapsed_seconds=monotonic() - started_at,
        results=ordered_results,
    )
    if print_summary:
        print(summary.format_overview())
        failure_summary = summary.format_failure_summary()
        if failure_summary:
            print(failure_summary)
    if summary.failed and not allow_partial:
        raise RuntimeError(summary.format_failure_summary() or "Reference data sync completed with failures.")
    if not summary.fundamental_paths and symbols:
        raise RuntimeError("Reference data sync failed for all requested symbols.")
    return summary


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
            "TOTAL_PARENT_EQUITY": "total_parent_equity",
            "TOTAL_EQUITY": "total_equity_fallback",
            "SHARE_CAPITAL": "share_capital",
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
        "total_parent_equity",
        "share_capital",
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
    merged["asset_growth"] = merged["total_assets"] / lag_assets.replace(0.0, np.nan) - 1.0
    merged["investment_to_assets"] = merged["capex_ttm"] / lag_assets.replace(0.0, np.nan)
    merged["accruals"] = (merged["parent_netprofit_ttm"] - merged["netcash_operate_ttm"]) / lag_assets.replace(
        0.0, np.nan
    )

    return merged[
        [
            "report_date",
            "available_date",
            "total_assets",
            "total_parent_equity",
            "share_capital",
            "total_operate_income_ttm",
            "operate_cost_ttm",
            "operate_profit_ttm",
            "parent_netprofit_ttm",
            "netcash_operate_ttm",
            "asset_growth",
            "investment_to_assets",
            "accruals",
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
    overwrite: bool,
) -> str:
    target = Path(reference_root) / "ashare" / "index" / f"{benchmark_symbol}.csv"
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date, date_column="timestamp"):
        return str(target)

    raw = _call_with_retries(
        lambda: ak.stock_zh_index_daily_em(
            symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
        ),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    normalized = raw.rename(columns={"date": "timestamp"}).copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target, index=False)
    _write_reference_sync_metadata(target, end_date=end_date)
    return str(target)


def _sync_fundamental_frame(
    *,
    ak,
    symbol: str,
    end_date: str,
    reference_root: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    overwrite: bool,
) -> ReferenceSymbolSyncResult:
    target = Path(reference_root) / "ashare" / "fundamentals" / f"{symbol}.csv"
    if target.exists() and not overwrite and _reference_sync_is_fresh(target, end_date=end_date):
        return ReferenceSymbolSyncResult(
            symbol=str(symbol),
            path=str(target),
            skipped=True,
            attempts=0,
        )

    em_symbol = _to_eastmoney_symbol(symbol)
    balance = _call_with_retries(
        lambda: ak.stock_balance_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    profit = _call_with_retries(
        lambda: ak.stock_profit_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    cash = _call_with_retries(
        lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=em_symbol),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    standardized = build_standardized_fundamental_frame(
        balance_frame=balance,
        profit_frame=profit,
        cash_flow_frame=cash,
    )
    standardized.insert(0, "symbol", str(symbol))
    target.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(target, index=False)
    _write_reference_sync_metadata(target, end_date=end_date)
    return ReferenceSymbolSyncResult(
        symbol=str(symbol),
        path=str(target),
        skipped=False,
        attempts=1,
    )


def _call_with_retries(fn, *, max_attempts: int, retry_backoff_seconds: float):
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_attempts and retry_backoff_seconds > 0:
                sleep(retry_backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def _reference_sync_is_fresh(target: Path, *, end_date: str, date_column: str | None = None) -> bool:
    metadata = _load_reference_sync_metadata(target)
    if metadata is not None:
        synced_end_date = metadata.get("synced_end_date")
        if synced_end_date:
            return pd.Timestamp(str(synced_end_date)) >= pd.Timestamp(end_date)
    if date_column is None:
        return False

    frame = pd.read_csv(target)
    candidate_columns = [date_column] if date_column in frame.columns else ["timestamp", "date", "available_date", "report_date"]
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty and values.max() >= pd.Timestamp(end_date):
            _write_reference_sync_metadata(target, end_date=end_date)
            return True
    return False


def _reference_sync_metadata_path(target: Path) -> Path:
    return target.with_suffix(f"{target.suffix}.sync.json")


def _load_reference_sync_metadata(target: Path) -> dict[str, object] | None:
    metadata_path = _reference_sync_metadata_path(target)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_reference_sync_metadata(target: Path, *, end_date: str) -> None:
    metadata_path = _reference_sync_metadata_path(target)
    payload = {
        "synced_end_date": pd.Timestamp(end_date).date().isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
