from __future__ import annotations

import numpy as np
import pandas as pd

from ml.factors.auxiliary import augment_factor_input_frame
from ml.factors.registry import FactorRegistry, get_default_registry


REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def build_factor_frame(
    data: pd.DataFrame,
    factor_names: list[str],
    *,
    registry: FactorRegistry | None = None,
    reference_root: str = "data/reference",
    market: str = "ashare",
) -> pd.DataFrame:
    active_registry = registry or get_default_registry()
    _validate_input_frame(data)
    if data.empty:
        return data.copy()

    ordered = data.sort_values(["symbol", "timestamp"]).copy()
    specs = [active_registry.get_factor(factor_name) for factor_name in factor_names]
    ordered = augment_factor_input_frame(
        data=ordered,
        specs=specs,
        reference_root=reference_root,
        market=market,
    )
    frames: list[pd.DataFrame] = []
    for _, frame in ordered.groupby("symbol", sort=False):
        item = frame.copy()
        cache: dict[tuple[str, int] | str, pd.Series] = {}
        for factor_name in factor_names:
            spec = active_registry.get_factor(factor_name)
            item[factor_name] = _compute_factor(item=item, kind=str(spec.params["kind"]), params=spec.params, cache=cache)
        item = _replace_infinite_values(item)
        item["history_count"] = item.reset_index().index + 1
        frames.append(item)

    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _validate_input_frame(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")


def _compute_factor(item: pd.DataFrame, *, kind: str, params: dict[str, object], cache: dict) -> pd.Series:
    if kind == "return":
        return _close_pct_change(item, int(params["window"]), cache)
    if kind == "momentum_skip":
        window = int(params["window"])
        skip_window = int(params["skip_window"])
        return item["close"].shift(skip_window) / item["close"].shift(window).replace(0.0, np.nan) - 1.0
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
        amount_proxy = item["amount"]
        if amount_proxy.isna().all():
            amount_proxy = item["close"] * item["volume"] * 100.0
        else:
            amount_proxy = amount_proxy.fillna(item["close"] * item["volume"] * 100.0)
        daily_illiquidity = returns / amount_proxy.replace(0.0, np.nan)
        return _rolling_mean(daily_illiquidity, window, cache, key=f"amihud_{window}")
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
    if kind == "gross_profit_to_assets":
        gross_profit = _fundamental_series(item, "fund_total_operate_income_ttm") - _fundamental_series(
            item,
            "fund_operate_cost_ttm",
        )
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
    if kind == "asset_growth":
        return _fundamental_series(item, "fund_asset_growth")
    if kind == "investment_to_assets":
        return _fundamental_series(item, "fund_investment_to_assets")
    if kind == "accruals":
        return _fundamental_series(item, "fund_accruals")
    raise ValueError(f"Unsupported factor kind: {kind}")


def _close_pct_change(item: pd.DataFrame, periods: int, cache: dict) -> pd.Series:
    key = ("close_pct_change", periods)
    if key not in cache:
        cache[key] = item["close"].pct_change(periods)
    return cache[key]


def _rolling_mean(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_mean", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).mean()
    return cache[cache_key]


def _rolling_std(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_std", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).std(ddof=0)
    return cache[cache_key]


def _rolling_max(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_max", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).max()
    return cache[cache_key]


def _rolling_min(series: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_min", key)
    if cache_key not in cache:
        cache[cache_key] = series.rolling(window, min_periods=window).min()
    return cache[cache_key]


def _previous_close(item: pd.DataFrame, cache: dict) -> pd.Series:
    key = "previous_close"
    if key not in cache:
        cache[key] = item["close"].shift(1)
    return cache[key]


def _benchmark_return(item: pd.DataFrame, benchmark_symbol: str, cache: dict) -> pd.Series:
    key = f"benchmark_return_{benchmark_symbol}"
    if key not in cache:
        column = f"benchmark_return_1_{benchmark_symbol}"
        if column not in item.columns:
            raise ValueError(f"Missing benchmark column: {column}")
        cache[key] = pd.to_numeric(item[column], errors="coerce")
    return cache[key]


def _rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int, cache: dict, *, key: str) -> pd.Series:
    cache_key = ("rolling_beta", key)
    if cache_key not in cache:
        covariance = asset_returns.rolling(window, min_periods=window).cov(benchmark_returns)
        variance = benchmark_returns.rolling(window, min_periods=window).var(ddof=0)
        cache[cache_key] = covariance / variance.replace(0.0, np.nan)
    return cache[cache_key]


def _daily_turnover(item: pd.DataFrame, cache: dict) -> pd.Series:
    key = "daily_turnover"
    if key not in cache:
        share_capital = _fundamental_series(item, "fund_share_capital").replace(0.0, np.nan)
        cache[key] = (item["volume"] * 100.0) / share_capital
    return cache[key]


def _fundamental_series(item: pd.DataFrame, column: str) -> pd.Series:
    if column not in item.columns:
        raise ValueError(f"Missing fundamental column: {column}")
    return pd.to_numeric(item[column], errors="coerce")


def _compute_rsi(item: pd.DataFrame, *, window: int, cache: dict) -> pd.Series:
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
    numeric_columns = item.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        return item
    item.loc[:, numeric_columns] = item.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
    return item
