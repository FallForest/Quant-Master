from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from time import monotonic

import numpy as np
import pandas as pd

from data.thread_parallel import run_bounded_thread_pool
from ml.factors.auxiliary import augment_factor_input_frame
from ml.factors.registry import FactorRegistry, get_default_registry


REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


@dataclass(slots=True)
class FactorBuildResult:
    frame: pd.DataFrame
    diagnostics: dict[str, object]


def build_factor_frame(
    data: pd.DataFrame,
    factor_names: list[str],
    *,
    registry: FactorRegistry | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    max_workers: int | None = None,
) -> pd.DataFrame:
    return build_factor_frame_with_diagnostics(
        data=data,
        factor_names=factor_names,
        registry=registry,
        reference_root=reference_root,
        market=market,
        max_workers=max_workers,
    ).frame


def build_factor_frame_with_diagnostics(
    data: pd.DataFrame,
    factor_names: list[str],
    *,
    registry: FactorRegistry | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
    max_workers: int | None = None,
) -> FactorBuildResult:
    started_at = monotonic()
    active_registry = registry or get_default_registry()
    _validate_input_frame(data)
    if data.empty:
        return FactorBuildResult(
            frame=data.copy(),
            diagnostics={
                "symbol_count": 0,
                "factor_count": len(factor_names),
                "parallel": False,
                "requested_workers": int(max_workers) if max_workers is not None else None,
                "resolved_workers": 1,
                "reference_augment_seconds": 0.0,
                "symbol_compute_seconds": 0.0,
                "elapsed_seconds": 0.0,
            },
        )

    ordered = data.sort_values(["symbol", "timestamp"]).copy()
    specs = [active_registry.get_factor(factor_name) for factor_name in factor_names]
    augment_started_at = monotonic()
    ordered = augment_factor_input_frame(
        data=ordered,
        specs=specs,
        reference_root=reference_root,
        market=market,
        max_workers=max_workers,
    )
    reference_augment_seconds = monotonic() - augment_started_at
    grouped_frames = [frame for _, frame in ordered.groupby("symbol", sort=False)]
    resolved_workers = _resolve_factor_workers(max_workers=max_workers, symbol_count=len(grouped_frames))

    compute_started_at = monotonic()
    if resolved_workers <= 1:
        frames = [
            _build_symbol_factor_frame(
                frame=frame,
                factor_names=factor_names,
                registry=active_registry,
            )
            for frame in grouped_frames
        ]
    else:
        report = run_bounded_thread_pool(
            items=grouped_frames,
            max_workers=resolved_workers,
            submitter=lambda frame: (
                lambda: _build_symbol_factor_frame(
                    frame=frame,
                    factor_names=factor_names,
                    registry=active_registry,
                ),
                None,
            ),
        )
        frames = [outcome.value for outcome in report.outcomes if outcome.value is not None]
    symbol_compute_seconds = monotonic() - compute_started_at

    return FactorBuildResult(
        frame=pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True),
        diagnostics={
            "symbol_count": len(grouped_frames),
            "factor_count": len(factor_names),
            "parallel": resolved_workers > 1,
            "requested_workers": int(max_workers) if max_workers is not None else None,
            "resolved_workers": resolved_workers,
            "reference_augment_seconds": reference_augment_seconds,
            "symbol_compute_seconds": symbol_compute_seconds,
            "elapsed_seconds": monotonic() - started_at,
        },
    )


def _build_symbol_factor_frame(
    *,
    frame: pd.DataFrame,
    factor_names: list[str],
    registry: FactorRegistry,
) -> pd.DataFrame:
    item = frame.copy()
    cache: dict[tuple[str, int] | str, pd.Series] = {}
    factor_columns: dict[str, pd.Series] = {}
    for factor_name in factor_names:
        spec = registry.get_factor(factor_name)
        factor_columns[factor_name] = _compute_factor(
            item=item,
            kind=str(spec.params["kind"]),
            params=spec.params,
            cache=cache,
        )

    factor_frame = pd.DataFrame(factor_columns, index=item.index)
    history_frame = pd.DataFrame(
        {
            "history_count": np.arange(1, len(item) + 1, dtype=np.int64),
        },
        index=item.index,
    )
    combined = pd.concat([item, factor_frame, history_frame], axis=1)
    return _replace_infinite_values(combined)


def _resolve_factor_workers(*, max_workers: int | None, symbol_count: int) -> int:
    if symbol_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), symbol_count))
    detected = cpu_count() or 1
    return max(1, min(detected, symbol_count))


def _validate_input_frame(data: pd.DataFrame) -> None:
    """校验最基础的行情列是否齐全。"""

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")


def _compute_factor(item: pd.DataFrame, *, kind: str, params: dict[str, object], cache: dict) -> pd.Series:
    """按因子 kind 分发到具体公式。"""

    if kind == "return":
        return _close_pct_change(item, int(params["window"]), cache)
    if kind == "momentum_skip":
        window = int(params["window"])
        skip_window = int(params["skip_window"])
        return item["close"].shift(skip_window) / item["close"].shift(window).replace(0.0, np.nan) - 1.0
    if kind == "industry_momentum_skip":
        window = int(params["window"])
        skip_window = int(params["skip_window"])
        source_column = str(params.get("source_column", "industry_return_1_level_1"))
        if source_column not in item.columns:
            raise ValueError(f"Missing industry return column: {source_column}")
        return _compounded_return_from_series(
            pd.to_numeric(item[source_column], errors="coerce"),
            window=window,
            skip_window=skip_window,
            cache=cache,
            key=f"{source_column}_{window}_{skip_window}",
        )
    if kind == "short_term_reversal":
        return -_close_pct_change(item, int(params["window"]), cache)
    if kind == "sma_distance":
        window = int(params["window"])
        rolling_mean = _rolling_mean(item["close"], window, cache, key=f"close_mean_{window}")
        return item["close"] / rolling_mean.replace(0.0, np.nan) - 1.0
    if kind == "price_to_rolling_high":
        window = int(params["window"])
        rolling_high = _rolling_max(item["close"], window, cache, key=f"close_max_{window}")
        return item["close"] / rolling_high.replace(0.0, np.nan) - 1.0
    if kind == "price_to_rolling_low":
        window = int(params["window"])
        rolling_low = _rolling_min(item["close"], window, cache, key=f"close_min_{window}")
        return item["close"] / rolling_low.replace(0.0, np.nan) - 1.0
    if kind == "volatility":
        returns = _close_pct_change(item, 1, cache)
        window = int(params["window"])
        return _rolling_std(returns, window, cache, key=f"return_std_{window}")
    if kind == "volume_ratio":
        window = int(params["window"])
        volume_mean = _rolling_mean(item["volume"], window, cache, key=f"volume_mean_{window}")
        return item["volume"] / volume_mean.replace(0.0, np.nan) - 1.0
    if kind == "volume_trend":
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        short_volume = _rolling_mean(item["volume"], short_window, cache, key=f"volume_mean_{short_window}")
        long_volume = _rolling_mean(item["volume"], long_window, cache, key=f"volume_mean_{long_window}")
        return short_volume / long_volume.replace(0.0, np.nan) - 1.0
    if kind == "channel_position":
        window = int(params["window"])
        rolling_high = _rolling_max(item["high"], window, cache, key=f"high_max_{window}")
        rolling_low = _rolling_min(item["low"], window, cache, key=f"low_min_{window}")
        channel_width = (rolling_high - rolling_low).replace(0.0, np.nan)
        return (item["close"] - rolling_low) / channel_width
    if kind == "distance_to_high":
        window = int(params["window"])
        rolling_high = _rolling_max(item["high"], window, cache, key=f"high_max_{window}")
        return item["close"] / rolling_high.replace(0.0, np.nan) - 1.0
    if kind == "distance_to_low":
        window = int(params["window"])
        rolling_low = _rolling_min(item["low"], window, cache, key=f"low_min_{window}")
        return item["close"] / rolling_low.replace(0.0, np.nan) - 1.0
    if kind == "rsi":
        return _compute_rsi(item, window=int(params["window"]), cache=cache)
    if kind == "bollinger_zscore":
        window = int(params["window"])
        rolling_mean = _rolling_mean(item["close"], window, cache, key=f"close_mean_{window}")
        rolling_std = _rolling_std(item["close"], window, cache, key=f"close_std_{window}")
        return (item["close"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
    if kind == "stochastic_k":
        window = int(params["window"])
        rolling_high = _rolling_max(item["high"], window, cache, key=f"stoch_high_max_{window}")
        rolling_low = _rolling_min(item["low"], window, cache, key=f"stoch_low_min_{window}")
        channel_width = (rolling_high - rolling_low).replace(0.0, np.nan)
        return (item["close"] - rolling_low) / channel_width
    if kind == "stochastic_d":
        window = int(params["window"])
        d_window = int(params["d_window"])
        stochastic_k = _compute_factor(
            item=item,
            kind="stochastic_k",
            params={"window": window},
            cache=cache,
        )
        return _rolling_mean(stochastic_k, d_window, cache, key=f"stochastic_d_{window}_{d_window}")
    if kind == "williams_r":
        window = int(params["window"])
        rolling_high = _rolling_max(item["high"], window, cache, key=f"wr_high_max_{window}")
        rolling_low = _rolling_min(item["low"], window, cache, key=f"wr_low_min_{window}")
        channel_width = (rolling_high - rolling_low).replace(0.0, np.nan)
        return -100.0 * (rolling_high - item["close"]) / channel_width
    if kind == "money_flow_index":
        window = int(params["window"])
        typical_price = _typical_price(item, cache)
        money_flow = typical_price * item["volume"]
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0.0, 0.0)
        negative_flow = money_flow.where(price_diff < 0.0, 0.0).abs()
        positive_sum = _rolling_mean(positive_flow, window, cache, key=f"mfi_pos_mean_{window}") * window
        negative_sum = _rolling_mean(negative_flow, window, cache, key=f"mfi_neg_mean_{window}") * window
        money_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + money_ratio))
        return mfi.where(negative_sum.ne(0.0), 100.0)
    if kind == "intraday_range_pct":
        return (item["high"] - item["low"]) / item["close"].replace(0.0, np.nan)
    if kind == "open_to_close_pct":
        return (item["close"] - item["open"]) / item["open"].replace(0.0, np.nan)
    if kind == "overnight_gap_pct":
        previous_close = _previous_close(item, cache)
        return item["open"] / previous_close.replace(0.0, np.nan) - 1.0
    if kind == "close_location_value":
        price_range = (item["high"] - item["low"]).replace(0.0, np.nan)
        return (item["close"] - item["low"]) / price_range
    if kind == "high_to_close_pct":
        return item["high"] / item["close"].replace(0.0, np.nan) - 1.0
    if kind == "low_to_close_pct":
        return item["low"] / item["close"].replace(0.0, np.nan) - 1.0
    if kind == "upper_shadow_pct":
        candle_top = pd.concat([item["open"], item["close"]], axis=1).max(axis=1)
        return (item["high"] - candle_top) / item["close"].replace(0.0, np.nan)
    if kind == "lower_shadow_pct":
        candle_bottom = pd.concat([item["open"], item["close"]], axis=1).min(axis=1)
        return (candle_bottom - item["low"]) / item["close"].replace(0.0, np.nan)
    if kind == "real_body_pct":
        return (item["close"] - item["open"]).abs() / item["open"].replace(0.0, np.nan)
    if kind == "beta":
        window = int(params["window"])
        benchmark_symbol = str(params["benchmark_symbol"])
        asset_returns = _close_pct_change(item, 1, cache)
        benchmark_returns = _benchmark_return(item, benchmark_symbol, cache)
        return _rolling_beta(asset_returns, benchmark_returns, window, cache, key=f"beta_{window}_{benchmark_symbol}")
    if kind == "idiosyncratic_volatility":
        window = int(params["window"])
        benchmark_symbol = str(params["benchmark_symbol"])
        asset_returns = _close_pct_change(item, 1, cache)
        benchmark_returns = _benchmark_return(item, benchmark_symbol, cache)
        beta = _rolling_beta(asset_returns, benchmark_returns, window, cache, key=f"beta_{window}_{benchmark_symbol}")
        mean_asset = _rolling_mean(asset_returns, window, cache, key=f"asset_return_mean_{window}")
        mean_benchmark = _rolling_mean(benchmark_returns, window, cache, key=f"benchmark_return_mean_{window}_{benchmark_symbol}")
        alpha = mean_asset - beta * mean_benchmark
        residual = asset_returns - (alpha + beta * benchmark_returns)
        return _rolling_std(residual, window, cache, key=f"idio_vol_{window}_{benchmark_symbol}")
    if kind == "amihud":
        window = int(params["window"])
        returns = _close_pct_change(item, 1, cache).abs()
        amount_proxy = _amount_proxy(item, cache)
        daily_illiquidity = returns / amount_proxy.replace(0.0, np.nan)
        return _rolling_mean(daily_illiquidity, window, cache, key=f"amihud_{window}")
    if kind == "dollar_volume":
        window = int(params["window"])
        return _rolling_mean(_amount_proxy(item, cache), window, cache, key=f"dollar_volume_{window}")
    if kind == "turnover":
        window = int(params["window"])
        daily_turnover = _daily_turnover(item, cache)
        return _rolling_mean(daily_turnover, window, cache, key=f"turnover_mean_{window}")
    if kind == "abnormal_turnover":
        window = int(params["window"])
        daily_turnover = _daily_turnover(item, cache)
        rolling_turnover = _rolling_mean(daily_turnover, window, cache, key=f"turnover_mean_{window}")
        return daily_turnover / rolling_turnover.replace(0.0, np.nan) - 1.0
    if kind == "log_total_mkt_cap":
        market_cap = item["close"] * _fundamental_series(item, "fund_share_capital")
        return np.log(market_cap.replace(0.0, np.nan))
    if kind == "book_to_market":
        book_value_per_share = _fundamental_series(item, "fund_total_parent_equity") / _fundamental_series(
            item,
            "fund_share_capital",
        ).replace(0.0, np.nan)
        return book_value_per_share / item["close"].replace(0.0, np.nan)
    if kind == "earnings_to_price":
        earnings_per_share = _fundamental_series(item, "fund_parent_netprofit_ttm") / _fundamental_series(
            item,
            "fund_share_capital",
        ).replace(0.0, np.nan)
        return earnings_per_share / item["close"].replace(0.0, np.nan)
    if kind == "sales_to_price":
        sales_per_share = _fundamental_series(item, "fund_total_operate_income_ttm") / _fundamental_series(
            item,
            "fund_share_capital",
        ).replace(0.0, np.nan)
        return sales_per_share / item["close"].replace(0.0, np.nan)
    if kind == "cashflow_to_price":
        cashflow_per_share = _fundamental_series(item, "fund_netcash_operate_ttm") / _fundamental_series(
            item,
            "fund_share_capital",
        ).replace(0.0, np.nan)
        return cashflow_per_share / item["close"].replace(0.0, np.nan)
    if kind == "dividend_yield_ttm":
        if "dividend_cash_per_share_ttm" not in item.columns:
            raise ValueError("Missing dividend column: dividend_cash_per_share_ttm")
        return pd.to_numeric(item["dividend_cash_per_share_ttm"], errors="coerce") / item["close"].replace(0.0, np.nan)
    if kind == "gross_profit_to_assets":
        gross_profit = _gross_profit(item)
        return gross_profit / _fundamental_series(item, "fund_total_assets").replace(0.0, np.nan)
    if kind == "operating_profitability":
        return _fundamental_series(item, "fund_operate_profit_ttm") / _fundamental_series(
            item,
            "fund_total_parent_equity",
        ).replace(0.0, np.nan)
    if kind == "roe_ttm":
        return _fundamental_series(item, "fund_parent_netprofit_ttm") / _fundamental_series(
            item,
            "fund_total_parent_equity",
        ).replace(0.0, np.nan)
    if kind == "roa_ttm":
        return _fundamental_series(item, "fund_parent_netprofit_ttm") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "gross_margin":
        return _gross_profit(item) / _fundamental_series(item, "fund_total_operate_income_ttm").replace(0.0, np.nan)
    if kind == "operating_margin":
        return _fundamental_series(item, "fund_operate_profit_ttm") / _fundamental_series(
            item,
            "fund_total_operate_income_ttm",
        ).replace(0.0, np.nan)
    if kind == "cash_profitability":
        return _fundamental_series(item, "fund_netcash_operate_ttm") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "asset_turnover":
        return _fundamental_series(item, "fund_total_operate_income_ttm") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "dividend_payout_ratio_ttm":
        if "dividend_cash_per_share_ttm" not in item.columns:
            raise ValueError("Missing dividend column: dividend_cash_per_share_ttm")
        earnings_per_share = _fundamental_series(item, "fund_parent_netprofit_ttm") / _fundamental_series(
            item,
            "fund_share_capital",
        ).replace(0.0, np.nan)
        return pd.to_numeric(item["dividend_cash_per_share_ttm"], errors="coerce") / earnings_per_share.replace(0.0, np.nan)
    if kind == "liabilities_to_assets":
        return _fundamental_series(item, "fund_total_liabilities") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "liabilities_to_equity":
        return _fundamental_series(item, "fund_total_liabilities") / _fundamental_series(
            item,
            "fund_total_parent_equity",
        ).replace(0.0, np.nan)
    if kind == "cash_to_assets":
        return _fundamental_series(item, "fund_monetary_funds") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "inventory_to_assets":
        return _fundamental_series(item, "fund_inventory") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "receivables_to_assets":
        return _fundamental_series(item, "fund_accounts_rece") / _fundamental_series(
            item,
            "fund_total_assets",
        ).replace(0.0, np.nan)
    if kind == "asset_growth":
        return _fundamental_series(item, "fund_asset_growth")
    if kind == "investment_to_assets":
        return _fundamental_series(item, "fund_investment_to_assets")
    if kind == "accruals":
        return _fundamental_series(item, "fund_accruals")
    if kind == "inventory_growth":
        return _fundamental_series(item, "fund_inventory_growth")
    if kind == "receivables_growth":
        return _fundamental_series(item, "fund_receivables_growth")
    if kind == "capex_growth":
        return _fundamental_series(item, "fund_capex_growth")
    if kind == "downside_volatility":
        returns = _close_pct_change(item, 1, cache)
        window = int(params["window"])
        negative_returns = returns.clip(upper=0.0)
        squared_negative_returns = negative_returns.pow(2)
        return np.sqrt(_rolling_mean(squared_negative_returns, window, cache, key=f"downside_var_{window}"))
    if kind == "atr_pct":
        window = int(params["window"])
        true_range = _true_range(item, cache)
        return _rolling_mean(true_range, window, cache, key=f"atr_{window}") / item["close"].replace(0.0, np.nan)
    if kind == "parkinson_volatility":
        window = int(params["window"])
        log_hl = np.log(item["high"] / item["low"].replace(0.0, np.nan))
        estimator = log_hl.pow(2) / (4.0 * np.log(2.0))
        return np.sqrt(_rolling_mean(estimator, window, cache, key=f"parkinson_{window}"))
    if kind == "garman_klass_volatility":
        window = int(params["window"])
        log_hl = np.log(item["high"] / item["low"].replace(0.0, np.nan))
        log_co = np.log(item["close"] / item["open"].replace(0.0, np.nan))
        estimator = 0.5 * log_hl.pow(2) - ((2.0 * np.log(2.0)) - 1.0) * log_co.pow(2)
        return np.sqrt(_rolling_mean(estimator.clip(lower=0.0), window, cache, key=f"gk_{window}"))
    if kind == "qlib_price_ratio":
        field = str(params["field"])
        lag = int(params.get("lag", 0) or 0)
        source = _price_field_series(item, field=field, cache=cache)
        if field == "volume":
            current_volume = pd.to_numeric(item["volume"], errors="coerce")
            shifted = source.shift(lag) if lag else source
            return shifted / (current_volume + 1e-12)
        shifted = source.shift(lag) if lag else source
        return shifted / item["close"].replace(0.0, np.nan)
    if kind == "qlib_kbar":
        return _compute_qlib_kbar(item=item, feature=str(params["feature"]), cache=cache)
    if kind == "qlib_rolling":
        return _compute_qlib_rolling(item=item, op=str(params["op"]), window=int(params["window"]), cache=cache)
    raise ValueError(f"Unsupported factor kind: {kind}")


def _close_pct_change(item: pd.DataFrame, periods: int, cache: dict) -> pd.Series:
    """缓存 close 的收益率序列。"""

    key = ("close_pct_change", periods)
    if key not in cache:
        cache[key] = item["close"].pct_change(periods)
    return cache[key]


def _rolling_mean(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    """缓存滚动均值。"""

    cache_key = ("rolling_mean", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).mean()
    return cache[cache_key]


def _rolling_std(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    """缓存滚动标准差。"""

    cache_key = ("rolling_std", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).std(ddof=0)
    return cache[cache_key]


def _rolling_max(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    """缓存滚动最大值。"""

    cache_key = ("rolling_max", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).max()
    return cache[cache_key]


def _rolling_min(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    """缓存滚动最小值。"""

    cache_key = ("rolling_min", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).min()
    return cache[cache_key]


def _previous_close(item: pd.DataFrame, cache: dict) -> pd.Series:
    """缓存前一日收盘价。"""

    key = "previous_close"
    if key not in cache:
        cache[key] = item["close"].shift(1)
    return cache[key]


def _true_range(item: pd.DataFrame, cache: dict) -> pd.Series:
    key = "true_range"
    if key not in cache:
        previous_close = _previous_close(item, cache)
        ranges = pd.concat(
            [
                item["high"] - item["low"],
                (item["high"] - previous_close).abs(),
                (item["low"] - previous_close).abs(),
            ],
            axis=1,
        )
        cache[key] = ranges.max(axis=1)
    return cache[key]


def _typical_price(item: pd.DataFrame, cache: dict) -> pd.Series:
    key = "typical_price"
    if key not in cache:
        cache[key] = (item["high"] + item["low"] + item["close"]) / 3.0
    return cache[key]


def _compounded_return_from_series(
    series: pd.Series,
    *,
    window: int,
    skip_window: int,
    cache: dict,
    key: str,
) -> pd.Series:
    """从一条日收益序列构造跳空动量类复合收益。"""

    cache_key = ("compounded_return", key)
    if cache_key not in cache:
        effective_window = max(1, window - skip_window)
        shifted = (1.0 + series).shift(skip_window)
        cache[cache_key] = shifted.rolling(effective_window, min_periods=effective_window).apply(np.prod, raw=True) - 1.0
    return cache[cache_key]


def _benchmark_return(item: pd.DataFrame, benchmark_symbol: str, cache: dict) -> pd.Series:
    """读取并缓存基准指数收益率。"""

    key = f"benchmark_return_{benchmark_symbol}"
    if key not in cache:
        column = f"benchmark_return_1_{benchmark_symbol}"
        if column not in item.columns:
            raise ValueError(f"Missing benchmark column: {column}")
        cache[key] = pd.to_numeric(item[column], errors="coerce")
    return cache[key]


def _rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    """计算并缓存滚动 beta。"""

    cache_key = ("rolling_beta", key)
    if cache_key not in cache:
        covariance = asset_returns.rolling(window, min_periods=window).cov(benchmark_returns)
        variance = benchmark_returns.rolling(window, min_periods=window).var(ddof=0)
        cache[cache_key] = covariance / variance.replace(0.0, np.nan)
    return cache[cache_key]


def _daily_turnover(item: pd.DataFrame, cache: dict) -> pd.Series:
    """用成交量和股本近似计算日换手率。"""

    key = "daily_turnover"
    if key not in cache:
        share_capital = _fundamental_series(item, "fund_share_capital").replace(0.0, np.nan)
        cache[key] = (item["volume"] * 100.0) / share_capital
    return cache[key]


def _amount_proxy(item: pd.DataFrame, cache: dict) -> pd.Series:
    """统一成交额口径。"""

    key = "amount_proxy"
    if key not in cache:
        if "amount" in item.columns:
            amount_proxy = pd.to_numeric(item["amount"], errors="coerce")
            fallback = item["close"] * item["volume"] * 100.0
            cache[key] = amount_proxy.fillna(fallback)
        else:
            cache[key] = item["close"] * item["volume"] * 100.0
    return cache[key]


def _fundamental_series(item: pd.DataFrame, column: str) -> pd.Series:
    """读取并数值化财务列。"""

    if column not in item.columns:
        raise ValueError(f"Missing fundamental column: {column}")
    return pd.to_numeric(item[column], errors="coerce")


def _gross_profit(item: pd.DataFrame) -> pd.Series:
    """计算毛利润。"""

    return _fundamental_series(item, "fund_total_operate_income_ttm") - _fundamental_series(
        item,
        "fund_operate_cost_ttm",
    )


def _compute_rsi(item: pd.DataFrame, *, window: int, cache: dict) -> pd.Series:
    """计算 RSI。"""

    delta_key = "close_delta"
    if delta_key not in cache:
        cache[delta_key] = item["close"].diff()
    delta = cache[delta_key]
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = _rolling_mean(gain, window, cache, key=f"gain_mean_{window}")
    average_loss = _rolling_mean(loss, window, cache, key=f"loss_mean_{window}")
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(average_loss.ne(0.0), 100.0)
    rsi = rsi.where(average_gain.ne(0.0), 0.0)
    return rsi


def _replace_infinite_values(item: pd.DataFrame) -> pd.DataFrame:
    """把数值列中的正负无穷统一替换成 NaN。"""

    numeric_columns = item.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        return item
    item.loc[:, numeric_columns] = item.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    return item


def _price_field_series(item: pd.DataFrame, *, field: str, cache: dict) -> pd.Series:
    key = f"price_field_{field}"
    if key not in cache:
        if field == "vwap":
            cache[key] = _vwap(item, cache)
        else:
            if field not in item.columns:
                raise ValueError(f"Missing price column: {field}")
            cache[key] = pd.to_numeric(item[field], errors="coerce")
    return cache[key]


def _vwap(item: pd.DataFrame, cache: dict) -> pd.Series:
    key = "vwap"
    if key not in cache:
        if "vwap" in item.columns:
            cache[key] = pd.to_numeric(item["vwap"], errors="coerce")
        elif "amount" in item.columns:
            amount = pd.to_numeric(item["amount"], errors="coerce")
            cache[key] = amount / (pd.to_numeric(item["volume"], errors="coerce").replace(0.0, np.nan) * 100.0)
        else:
            cache[key] = _typical_price(item, cache)
    return cache[key]


def _compute_qlib_kbar(*, item: pd.DataFrame, feature: str, cache: dict) -> pd.Series:
    open_price = pd.to_numeric(item["open"], errors="coerce")
    close_price = pd.to_numeric(item["close"], errors="coerce")
    high_price = pd.to_numeric(item["high"], errors="coerce")
    low_price = pd.to_numeric(item["low"], errors="coerce")
    upper_body = pd.concat([open_price, close_price], axis=1).max(axis=1)
    lower_body = pd.concat([open_price, close_price], axis=1).min(axis=1)
    open_denominator = open_price.replace(0.0, np.nan)
    range_denominator = (high_price - low_price).replace(0.0, np.nan) + 1e-12

    if feature == "KMID":
        return (close_price - open_price) / open_denominator
    if feature == "KLEN":
        return (high_price - low_price) / open_denominator
    if feature == "KMID2":
        return (close_price - open_price) / range_denominator
    if feature == "KUP":
        return (high_price - upper_body) / open_denominator
    if feature == "KUP2":
        return (high_price - upper_body) / range_denominator
    if feature == "KLOW":
        return (lower_body - low_price) / open_denominator
    if feature == "KLOW2":
        return (lower_body - low_price) / range_denominator
    if feature == "KSFT":
        return (2.0 * close_price - high_price - low_price) / open_denominator
    if feature == "KSFT2":
        return (2.0 * close_price - high_price - low_price) / range_denominator
    raise ValueError(f"Unsupported Qlib kbar feature: {feature}")


def _compute_qlib_rolling(*, item: pd.DataFrame, op: str, window: int, cache: dict) -> pd.Series:
    close_price = pd.to_numeric(item["close"], errors="coerce")
    high_price = pd.to_numeric(item["high"], errors="coerce")
    low_price = pd.to_numeric(item["low"], errors="coerce")
    volume = pd.to_numeric(item["volume"], errors="coerce")
    close_denominator = close_price.replace(0.0, np.nan)
    volume_denominator = volume + 1e-12

    if op == "ROC":
        return close_price.shift(window) / close_denominator
    if op == "MA":
        return _rolling_mean(close_price, window, cache, key=f"qlib_close_mean_{window}") / close_denominator
    if op == "STD":
        return _rolling_std(close_price, window, cache, key=f"qlib_close_std_{window}") / close_denominator
    if op == "BETA":
        return _rolling_linear_slope(close_price, window, cache, key=f"qlib_close_slope_{window}") / close_denominator
    if op == "RSQR":
        return _rolling_rsquare(close_price, window, cache, key=f"qlib_close_rsqr_{window}")
    if op == "RESI":
        return _rolling_residual(close_price, window, cache, key=f"qlib_close_resi_{window}") / close_denominator
    if op == "MAX":
        return _rolling_max(high_price, window, cache, key=f"qlib_high_max_{window}") / close_denominator
    if op == "MIN":
        return _rolling_min(low_price, window, cache, key=f"qlib_low_min_{window}") / close_denominator
    if op == "QTLU":
        return _rolling_quantile(close_price, window, 0.8, cache, key=f"qlib_close_qtlu_{window}") / close_denominator
    if op == "QTLD":
        return _rolling_quantile(close_price, window, 0.2, cache, key=f"qlib_close_qtld_{window}") / close_denominator
    if op == "RANK":
        return _rolling_percentile_rank(close_price, window, cache, key=f"qlib_close_rank_{window}")
    if op == "RSV":
        channel = (_rolling_max(high_price, window, cache, key=f"qlib_rsv_high_{window}") - _rolling_min(low_price, window, cache, key=f"qlib_rsv_low_{window}")).replace(0.0, np.nan)
        return (close_price - _rolling_min(low_price, window, cache, key=f"qlib_rsv_low_{window}")) / channel
    if op == "IMAX":
        return _rolling_days_since_extreme(high_price, window, mode="max", cache=cache, key=f"qlib_imax_{window}") / window
    if op == "IMIN":
        return _rolling_days_since_extreme(low_price, window, mode="min", cache=cache, key=f"qlib_imin_{window}") / window
    if op == "IMXD":
        max_days = _rolling_days_since_extreme(high_price, window, mode="max", cache=cache, key=f"qlib_imxd_max_{window}")
        min_days = _rolling_days_since_extreme(low_price, window, mode="min", cache=cache, key=f"qlib_imxd_min_{window}")
        return (max_days - min_days) / window
    if op == "CORR":
        return _rolling_corr(close_price, np.log(volume + 1.0), window, cache, key=f"qlib_corr_{window}")
    if op == "CORD":
        return _rolling_corr(
            close_price / close_price.shift(1).replace(0.0, np.nan),
            np.log(volume / volume.shift(1).replace(0.0, np.nan) + 1.0),
            window,
            cache,
            key=f"qlib_cord_{window}",
        )
    if op == "CNTP":
        return _rolling_mean((close_price > close_price.shift(1)).astype(float), window, cache, key=f"qlib_cntp_{window}")
    if op == "CNTN":
        return _rolling_mean((close_price < close_price.shift(1)).astype(float), window, cache, key=f"qlib_cntn_{window}")
    if op == "CNTD":
        up = _rolling_mean((close_price > close_price.shift(1)).astype(float), window, cache, key=f"qlib_cntd_up_{window}")
        down = _rolling_mean((close_price < close_price.shift(1)).astype(float), window, cache, key=f"qlib_cntd_down_{window}")
        return up - down
    if op == "SUMP":
        delta = close_price - close_price.shift(1)
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_sump_abs_{window}")
        return _rolling_sum(delta.clip(lower=0.0), window, cache, key=f"qlib_sump_pos_{window}") / (absolute_sum + 1e-12)
    if op == "SUMN":
        delta = close_price - close_price.shift(1)
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_sumn_abs_{window}")
        return _rolling_sum((-delta).clip(lower=0.0), window, cache, key=f"qlib_sumn_neg_{window}") / (absolute_sum + 1e-12)
    if op == "SUMD":
        delta = close_price - close_price.shift(1)
        positive = _rolling_sum(delta.clip(lower=0.0), window, cache, key=f"qlib_sumd_pos_{window}")
        negative = _rolling_sum((-delta).clip(lower=0.0), window, cache, key=f"qlib_sumd_neg_{window}")
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_sumd_abs_{window}")
        return (positive - negative) / (absolute_sum + 1e-12)
    if op == "VMA":
        return _rolling_mean(volume, window, cache, key=f"qlib_volume_mean_{window}") / volume_denominator
    if op == "VSTD":
        return _rolling_std(volume, window, cache, key=f"qlib_volume_std_{window}") / volume_denominator
    if op == "WVMA":
        weighted_move = (close_price / close_price.shift(1).replace(0.0, np.nan) - 1.0).abs() * volume
        numerator = _rolling_std(weighted_move, window, cache, key=f"qlib_wvma_std_{window}")
        denominator = _rolling_mean(weighted_move, window, cache, key=f"qlib_wvma_mean_{window}")
        return numerator / (denominator + 1e-12)
    if op == "VSUMP":
        delta = volume - volume.shift(1)
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_vsump_abs_{window}")
        return _rolling_sum(delta.clip(lower=0.0), window, cache, key=f"qlib_vsump_pos_{window}") / (absolute_sum + 1e-12)
    if op == "VSUMN":
        delta = volume - volume.shift(1)
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_vsumn_abs_{window}")
        return _rolling_sum((-delta).clip(lower=0.0), window, cache, key=f"qlib_vsumn_neg_{window}") / (absolute_sum + 1e-12)
    if op == "VSUMD":
        delta = volume - volume.shift(1)
        positive = _rolling_sum(delta.clip(lower=0.0), window, cache, key=f"qlib_vsumd_pos_{window}")
        negative = _rolling_sum((-delta).clip(lower=0.0), window, cache, key=f"qlib_vsumd_neg_{window}")
        absolute_sum = _rolling_sum(delta.abs(), window, cache, key=f"qlib_vsumd_abs_{window}")
        return (positive - negative) / (absolute_sum + 1e-12)
    raise ValueError(f"Unsupported Qlib rolling op: {op}")


def _rolling_sum(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_sum", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).sum()
    return cache[cache_key]


def _rolling_quantile(series: pd.Series, window: int, quantile: float, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_quantile", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).quantile(quantile)
    return cache[cache_key]


def _rolling_percentile_rank(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_percentile_rank", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).apply(
            lambda values: pd.Series(values).rank(pct=True).iloc[-1],
            raw=False,
        )
    return cache[cache_key]


def _rolling_days_since_extreme(series: pd.Series, window: int, *, mode: str, cache: dict, key: str) -> pd.Series:
    cache_key = ("rolling_days_since_extreme", key)
    if cache_key not in cache:
        if mode == "max":
            fn = lambda values: float(window - 1 - int(np.argmax(values)))
        elif mode == "min":
            fn = lambda values: float(window - 1 - int(np.argmin(values)))
        else:
            raise ValueError(f"Unsupported extreme mode: {mode}")
        cache[cache_key] = series.rolling(window, min_periods=window).apply(fn, raw=True)
    return cache[cache_key]


def _rolling_corr(left: pd.Series, right: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_corr", key)
    if cache_key not in cache:
        cache[cache_key] = left.rolling(window, min_periods=window).corr(right)
    return cache[cache_key]


def _rolling_linear_regression(series: pd.Series, window: int, cache: dict, *, key: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    cache_key = ("rolling_linear_regression", key)
    if cache_key not in cache:
        x = np.arange(window, dtype=float)

        def regression(values: np.ndarray) -> np.ndarray:
            y = np.asarray(values, dtype=float)
            if np.isnan(y).any():
                return np.array([np.nan, np.nan, np.nan], dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            fitted = intercept + slope * x
            residual = y[-1] - fitted[-1]
            denom = float(np.square(y - y.mean()).sum())
            rsquare = np.nan if denom == 0.0 else 1.0 - float(np.square(y - fitted).sum()) / denom
            return np.array([slope, rsquare, residual], dtype=float)

        # pandas rolling apply is scalar-only, so compute each metric separately.
        slope = series.rolling(window, min_periods=window).apply(lambda values: regression(values)[0], raw=True)
        rsquare = series.rolling(window, min_periods=window).apply(lambda values: regression(values)[1], raw=True)
        residual = series.rolling(window, min_periods=window).apply(lambda values: regression(values)[2], raw=True)
        cache[cache_key] = (slope, rsquare, residual)
    return cache[cache_key]


def _rolling_linear_slope(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    return _rolling_linear_regression(series, window, cache, key=key)[0]


def _rolling_rsquare(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    return _rolling_linear_regression(series, window, cache, key=key)[1]


def _rolling_residual(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    return _rolling_linear_regression(series, window, cache, key=key)[2]
