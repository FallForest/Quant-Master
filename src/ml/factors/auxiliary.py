from __future__ import annotations

from os import cpu_count
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.thread_parallel import run_bounded_thread_pool
from ml.factors.specs import FactorSpec


BENCHMARK_FACTOR_KINDS = {"beta", "idiosyncratic_volatility"}
INDUSTRY_FACTOR_KINDS = {"industry_momentum_skip"}
DIVIDEND_FACTOR_KINDS = {"dividend_yield_ttm", "dividend_payout_ratio_ttm"}
FUNDAMENTAL_FACTOR_KINDS = {
    "turnover",
    "abnormal_turnover",
    "log_total_mkt_cap",
    "book_to_market",
    "earnings_to_price",
    "sales_to_price",
    "cashflow_to_price",
    "dividend_payout_ratio_ttm",
    "gross_profit_to_assets",
    "operating_profitability",
    "roe_ttm",
    "roa_ttm",
    "gross_margin",
    "operating_margin",
    "cash_profitability",
    "asset_turnover",
    "liabilities_to_assets",
    "liabilities_to_equity",
    "cash_to_assets",
    "inventory_to_assets",
    "receivables_to_assets",
    "asset_growth",
    "investment_to_assets",
    "accruals",
    "inventory_growth",
    "receivables_growth",
    "capex_growth",
}

REQUIRED_FUNDAMENTAL_COLUMNS = [
    "available_date",
    "report_date",
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

OPTIONAL_FUNDAMENTAL_COLUMNS = [
    "total_liabilities",
    "inventory",
    "accounts_rece",
    "note_rece",
    "fixed_asset",
    "intangible_asset",
    "monetary_funds",
    "capex_ttm",
    "inventory_growth",
    "receivables_growth",
    "capex_growth",
]


def augment_factor_input_frame(
    data: pd.DataFrame,
    *,
    specs: list[FactorSpec],
    reference_root: str,
    market: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    factor_kinds = {str(spec.params["kind"]) for spec in specs}
    enriched = data.copy()

    if BENCHMARK_FACTOR_KINDS & factor_kinds:
        benchmark_symbols = sorted(
            {
                str(spec.params["benchmark_symbol"])
                for spec in specs
                if str(spec.params["kind"]) in BENCHMARK_FACTOR_KINDS
            }
        )
        for benchmark_symbol in benchmark_symbols:
            enriched = _merge_benchmark_frame(
                data=enriched,
                reference_root=reference_root,
                market=market,
                benchmark_symbol=benchmark_symbol,
            )

    if FUNDAMENTAL_FACTOR_KINDS & factor_kinds:
        enriched = _merge_fundamental_frames(
            data=enriched,
            reference_root=reference_root,
            market=market,
            max_workers=max_workers,
        )

    if INDUSTRY_FACTOR_KINDS & factor_kinds:
        enriched = _merge_industry_frames(
            data=enriched,
            reference_root=reference_root,
            market=market,
            max_workers=max_workers,
        )

    if DIVIDEND_FACTOR_KINDS & factor_kinds:
        enriched = _merge_dividend_frames(
            data=enriched,
            reference_root=reference_root,
            market=market,
            max_workers=max_workers,
        )

    return enriched


def _merge_benchmark_frame(
    *,
    data: pd.DataFrame,
    reference_root: str,
    market: str,
    benchmark_symbol: str,
) -> pd.DataFrame:
    benchmark_path = Path(reference_root) / market / "index" / f"{benchmark_symbol}.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Missing benchmark reference data: {benchmark_path}. "
            f"Run `py sync_ashare_reference_data.py --benchmark-symbol {benchmark_symbol}` first."
        )

    benchmark = pd.read_csv(benchmark_path)
    timestamp_column = "timestamp" if "timestamp" in benchmark.columns else "date"
    if timestamp_column not in benchmark.columns or "close" not in benchmark.columns:
        raise ValueError(f"Benchmark file is missing required columns: {benchmark_path}")

    normalized = benchmark.rename(columns={timestamp_column: "timestamp"}).copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    close_column = f"benchmark_close_{benchmark_symbol}"
    return_column = f"benchmark_return_1_{benchmark_symbol}"
    normalized = normalized[["timestamp", "close"]].rename(columns={"close": close_column})
    normalized[return_column] = normalized[close_column].pct_change()
    return data.merge(normalized, on="timestamp", how="left")


def _merge_fundamental_frames(
    *,
    data: pd.DataFrame,
    reference_root: str,
    market: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    return _map_symbol_frames(
        data=data,
        max_workers=max_workers,
        worker_fn=lambda symbol, item: _merge_single_fundamental_frame(
            symbol=symbol,
            item=item,
            reference_root=reference_root,
            market=market,
        ),
    )


def _merge_industry_frames(
    *,
    data: pd.DataFrame,
    reference_root: str,
    market: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    enriched = _map_symbol_frames(
        data=data,
        max_workers=max_workers,
        worker_fn=lambda symbol, item: _merge_single_industry_frame(
            symbol=symbol,
            item=item,
            reference_root=reference_root,
            market=market,
        ),
    )
    daily_returns = enriched.groupby("symbol", sort=False)["close"].pct_change()
    enriched["_self_return_1"] = pd.to_numeric(daily_returns, errors="coerce")
    industry_group = enriched.groupby(["timestamp", "industry_level_1"], dropna=False)["_self_return_1"]
    enriched["_industry_return_sum"] = industry_group.transform("sum")
    enriched["_industry_return_count"] = industry_group.transform(lambda value: int(value.notna().sum()))
    valid_peer_count = enriched["_industry_return_count"] - 1
    leave_one_out = (enriched["_industry_return_sum"] - enriched["_self_return_1"]) / valid_peer_count.replace(0, np.nan)
    enriched["industry_return_1_level_1"] = leave_one_out.where(valid_peer_count > 0)
    return enriched.drop(columns=["_self_return_1", "_industry_return_sum", "_industry_return_count"])


def _merge_dividend_frames(
    *,
    data: pd.DataFrame,
    reference_root: str,
    market: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    return _map_symbol_frames(
        data=data,
        max_workers=max_workers,
        worker_fn=lambda symbol, item: _merge_single_dividend_frame(
            symbol=symbol,
            item=item,
            reference_root=reference_root,
            market=market,
        ),
    )


def _map_symbol_frames(
    *,
    data: pd.DataFrame,
    max_workers: int | None,
    worker_fn: Callable[[str, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    symbol_frames = [(str(symbol), item.copy()) for symbol, item in data.groupby("symbol", sort=False)]
    if not symbol_frames:
        return data.copy()

    resolved_workers = _resolve_symbol_workers(max_workers=max_workers, symbol_count=len(symbol_frames))
    if resolved_workers <= 1:
        frames = [worker_fn(symbol, item) for symbol, item in symbol_frames]
    else:
        report = run_bounded_thread_pool(
            items=symbol_frames,
            max_workers=resolved_workers,
            submitter=lambda item: (lambda: worker_fn(str(item[0]), item[1]), None),
        )
        frames = [outcome.value for outcome in report.outcomes if outcome.value is not None]
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _merge_single_fundamental_frame(
    *,
    symbol: str,
    item: pd.DataFrame,
    reference_root: str,
    market: str,
) -> pd.DataFrame:
    fundamental_path = Path(reference_root) / market / "fundamentals" / f"{symbol}.csv"
    if not fundamental_path.exists():
        raise FileNotFoundError(
            f"Missing fundamental reference data: {fundamental_path}. "
            f"Run `py sync_ashare_reference_data.py --symbols {symbol}` first."
        )
    fundamentals = pd.read_csv(fundamental_path)
    missing = [column for column in REQUIRED_FUNDAMENTAL_COLUMNS if column not in fundamentals.columns]
    if missing:
        raise ValueError(f"Fundamental file missing required columns {missing}: {fundamental_path}")
    for column in OPTIONAL_FUNDAMENTAL_COLUMNS:
        if column not in fundamentals.columns:
            fundamentals[column] = pd.NA

    right = fundamentals.copy()
    if "symbol" in right.columns:
        right = right.drop(columns=["symbol"])
    right["available_date"] = pd.to_datetime(right["available_date"])
    right["report_date"] = pd.to_datetime(right["report_date"])
    prefixed = right.rename(columns={column: f"fund_{column}" for column in right.columns if column != "symbol"})
    return pd.merge_asof(
        item.sort_values("timestamp"),
        prefixed.sort_values("fund_available_date"),
        left_on="timestamp",
        right_on="fund_available_date",
        direction="backward",
    )


def _merge_single_industry_frame(
    *,
    symbol: str,
    item: pd.DataFrame,
    reference_root: str,
    market: str,
) -> pd.DataFrame:
    industry_path = Path(reference_root) / market / "industry" / f"{symbol}.csv"
    item = item.sort_values("timestamp").copy()
    if not industry_path.exists():
        item["industry_level_1"] = pd.NA
        return item

    reference = pd.read_csv(industry_path)
    if reference.empty or "change_date" not in reference.columns:
        item["industry_level_1"] = pd.NA
        return item

    reference["change_date"] = pd.to_datetime(reference["change_date"])
    active = reference.copy()
    if "symbol" in active.columns:
        active = active.drop(columns=["symbol"])
    merged = pd.merge_asof(
        item,
        active.sort_values("change_date"),
        left_on="timestamp",
        right_on="change_date",
        direction="backward",
    )
    merged["industry_level_1"] = merged["industry_level_1"].fillna(merged.get("sector"))
    return merged


def _merge_single_dividend_frame(
    *,
    symbol: str,
    item: pd.DataFrame,
    reference_root: str,
    market: str,
) -> pd.DataFrame:
    dividend_path = Path(reference_root) / market / "dividends" / f"{symbol}.csv"
    item = item.sort_values("timestamp").copy()
    if not dividend_path.exists():
        item["dividend_cash_per_share_ttm"] = pd.NA
        return item

    reference = pd.read_csv(dividend_path)
    if reference.empty or "event_date" not in reference.columns or "cash_dividend_per_share" not in reference.columns:
        item["dividend_cash_per_share_ttm"] = pd.NA
        return item

    events = reference.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["cash_dividend_per_share"] = pd.to_numeric(events["cash_dividend_per_share"], errors="coerce")
    events = events.dropna(subset=["event_date", "cash_dividend_per_share"]).sort_values("event_date").reset_index(drop=True)
    if events.empty:
        item["dividend_cash_per_share_ttm"] = pd.NA
        return item

    events["dividend_cumulative"] = events["cash_dividend_per_share"].cumsum()
    current = pd.merge_asof(
        item[["timestamp"]].sort_values("timestamp"),
        events[["event_date", "dividend_cumulative"]].sort_values("event_date"),
        left_on="timestamp",
        right_on="event_date",
        direction="backward",
    )["dividend_cumulative"].fillna(0.0)
    window_start = item[["timestamp"]].sort_values("timestamp").rename(columns={"timestamp": "window_start"})
    window_start["window_start"] = window_start["window_start"] - pd.Timedelta(days=365)
    prior = pd.merge_asof(
        window_start,
        events[["event_date", "dividend_cumulative"]].sort_values("event_date"),
        left_on="window_start",
        right_on="event_date",
        direction="backward",
    )["dividend_cumulative"].fillna(0.0)
    item["dividend_cash_per_share_ttm"] = (current - prior).to_numpy()
    return item


def _resolve_symbol_workers(*, max_workers: int | None, symbol_count: int) -> int:
    if symbol_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), symbol_count))
    detected = cpu_count() or 1
    return max(1, min(detected, symbol_count))
