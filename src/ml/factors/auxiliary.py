from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.factors.specs import FactorSpec

BENCHMARK_FACTOR_KINDS = {"beta", "idiosyncratic_volatility"}
FUNDAMENTAL_FACTOR_KINDS = {
    "turnover",
    "abnormal_turnover",
    "log_total_mkt_cap",
    "book_to_market",
    "earnings_to_price",
    "sales_to_price",
    "gross_profit_to_assets",
    "operating_profitability",
    "roe_ttm",
    "asset_growth",
    "investment_to_assets",
    "accruals",
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


def augment_factor_input_frame(
    data: pd.DataFrame,
    *,
    specs: list[FactorSpec],
    reference_root: str,
    market: str,
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
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, item in data.groupby("symbol", sort=False):
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

        right = fundamentals.copy()
        if "symbol" in right.columns:
            right = right.drop(columns=["symbol"])
        right["available_date"] = pd.to_datetime(right["available_date"])
        right["report_date"] = pd.to_datetime(right["report_date"])
        prefixed = right.rename(columns={column: f"fund_{column}" for column in right.columns if column != "symbol"})
        merged = pd.merge_asof(
            item.sort_values("timestamp"),
            prefixed.sort_values("fund_available_date"),
            left_on="timestamp",
            right_on="fund_available_date",
            direction="backward",
        )
        frames.append(merged)

    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
