from __future__ import annotations

from ml.factors.specs import FactorSpec


class FactorRegistry:
    def __init__(self, factors: list[FactorSpec]) -> None:
        self._factors: dict[str, FactorSpec] = {}
        for factor in factors:
            self.register(factor)

    def register(self, factor: FactorSpec) -> None:
        if factor.name in self._factors:
            raise ValueError(f"Duplicate factor registration: {factor.name}")
        self._factors[factor.name] = factor

    def get_factor(self, name: str) -> FactorSpec:
        try:
            return self._factors[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factors))
            raise ValueError(f"Unknown factor: {name}. Available factors: {available}") from exc

    def has_factor(self, name: str) -> bool:
        return name in self._factors

    def list_factors(self) -> list[FactorSpec]:
        return [self._factors[name] for name in sorted(self._factors)]

    def list_factor_names(self) -> list[str]:
        return sorted(self._factors)

    def list_factors_by_family(self, family: str) -> list[FactorSpec]:
        return [factor for factor in self.list_factors() if factor.family == family]


def estimate_factor_history_lookback(
    factor_names: list[str],
    *,
    registry: FactorRegistry | None = None,
) -> int:
    active_registry = registry or get_default_registry()
    lookbacks = [_estimate_factor_spec_history_lookback(active_registry.get_factor(name)) for name in factor_names]
    return max(lookbacks, default=0)


def _estimate_factor_spec_history_lookback(spec: FactorSpec) -> int:
    kind = str(spec.params.get("kind", "") or "")
    window = int(spec.params.get("window", 0) or 0)
    skip_window = int(spec.params.get("skip_window", 0) or 0)
    short_window = int(spec.params.get("short_window", 0) or 0)
    long_window = int(spec.params.get("long_window", 0) or 0)
    d_window = int(spec.params.get("d_window", 0) or 0)

    if kind in {
        "intraday_range_pct",
        "open_to_close_pct",
        "close_location_value",
        "high_to_close_pct",
        "low_to_close_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "real_body_pct",
    }:
        return 0
    if kind == "overnight_gap_pct":
        return 1
    if kind in {
        "return",
        "short_term_reversal",
        "sma_distance",
        "price_to_rolling_high",
        "price_to_rolling_low",
        "volume_ratio",
        "channel_position",
        "distance_to_high",
        "distance_to_low",
        "stochastic_k",
        "williams_r",
        "parkinson_volatility",
        "garman_klass_volatility",
    }:
        return window
    if kind in {"volatility", "beta", "idiosyncratic_volatility", "amihud", "downside_volatility", "atr_pct", "money_flow_index"}:
        return window + 1
    if kind == "volume_trend":
        return max(short_window, long_window)
    if kind == "rsi":
        return window + 1
    if kind == "stochastic_d":
        return max(0, window + d_window - 1)
    if kind == "bollinger_zscore":
        return window
    if kind in {"momentum_skip", "industry_momentum_skip"}:
        return max(window, skip_window)
    if kind in {
        "turnover",
        "abnormal_turnover",
        "dollar_volume",
    }:
        return window
    if kind in {
        "log_total_mkt_cap",
        "book_to_market",
        "earnings_to_price",
        "sales_to_price",
        "cashflow_to_price",
        "dividend_yield_ttm",
        "gross_profit_to_assets",
        "operating_profitability",
        "roe_ttm",
        "roa_ttm",
        "gross_margin",
        "operating_margin",
        "cash_profitability",
        "asset_turnover",
        "dividend_payout_ratio_ttm",
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
    }:
        return 0
    return max(window, short_window, long_window)


def _factor(
    *,
    name: str,
    family: str,
    description: str,
    params: dict[str, object],
    tags: list[str],
    stability_level: str = "stable",
) -> FactorSpec:
    return FactorSpec(
        name=name,
        family=family,
        version="v1",
        description=description,
        input_columns=["open", "high", "low", "close", "volume"],
        params=params,
        dependencies=[],
        tags=tags,
        stability_level=stability_level,
    )


DEFAULT_FACTOR_REGISTRY = FactorRegistry(
    factors=[
        _factor(
            name=f"return_{window}",
            family="trend",
            description=f"{window}-day close-to-close return.",
            params={"kind": "return", "window": window},
            tags=["price_only", "trend"],
            stability_level="stable" if window in {20} else "candidate",
        )
        for window in [1, 3, 5, 10, 20, 60, 120]
    ]
    + [
        _factor(
            name=f"close_to_sma_{window}",
            family="trend",
            description=f"Distance from close to {window}-day simple moving average.",
            params={"kind": "sma_distance", "window": window},
            tags=["price_only", "trend"],
            stability_level="stable" if window in {20} else "candidate",
        )
        for window in [5, 10, 20, 60, 120]
    ]
    + [
        _factor(
            name=f"volatility_{window}",
            family="volatility",
            description=f"{window}-day rolling volatility of daily returns.",
            params={"kind": "volatility", "window": window},
            tags=["risk"],
            stability_level="stable" if window in {20} else "candidate",
        )
        for window in [5, 10, 20, 60]
    ]
    + [
        _factor(
            name=f"volume_ratio_{window}",
            family="volume",
            description=f"Volume relative to its {window}-day rolling mean.",
            params={"kind": "volume_ratio", "window": window},
            tags=["volume"],
            stability_level="stable" if window in {20} else "candidate",
        )
        for window in [5, 20, 60]
    ]
    + [
        _factor(
            name="volume_trend_5_20",
            family="volume",
            description="Short-vs-medium volume acceleration using 5-day and 20-day averages.",
            params={"kind": "volume_trend", "short_window": 5, "long_window": 20},
            tags=["volume", "trend"],
        ),
        _factor(
            name="volume_trend_10_60",
            family="volume",
            description="Short-vs-long volume acceleration using 10-day and 60-day averages.",
            params={"kind": "volume_trend", "short_window": 10, "long_window": 60},
            tags=["volume", "trend"],
            stability_level="candidate",
        ),
        _factor(
            name="momentum_63_21",
            family="trend",
            description="3-1 momentum using the return from 63 trading days ago to 21 days ago.",
            params={"kind": "momentum_skip", "window": 63, "skip_window": 21},
            tags=["price_only", "trend", "momentum"],
            stability_level="candidate",
        ),
        _factor(
            name="momentum_126_21",
            family="trend",
            description="6-1 momentum using the return from 126 trading days ago to 21 days ago.",
            params={"kind": "momentum_skip", "window": 126, "skip_window": 21},
            tags=["price_only", "trend", "momentum"],
        ),
        _factor(
            name="momentum_252_21",
            family="trend",
            description="12-1 momentum using the return from 252 trading days ago to 21 days ago.",
            params={"kind": "momentum_skip", "window": 252, "skip_window": 21},
            tags=["price_only", "trend", "momentum"],
        ),
        _factor(
            name="industry_momentum_63_21",
            family="trend",
            description="3-1 industry momentum using equal-weighted same-industry returns from 63 trading days ago to 21 days ago.",
            params={"kind": "industry_momentum_skip", "window": 63, "skip_window": 21, "source_column": "industry_return_1_level_1"},
            tags=["industry", "trend", "momentum"],
            stability_level="candidate",
        ),
        _factor(
            name="industry_momentum_126_21",
            family="trend",
            description="6-1 industry momentum using equal-weighted same-industry returns from 126 trading days ago to 21 days ago.",
            params={"kind": "industry_momentum_skip", "window": 126, "skip_window": 21, "source_column": "industry_return_1_level_1"},
            tags=["industry", "trend", "momentum"],
            stability_level="candidate",
        ),
        _factor(
            name="industry_momentum_252_21",
            family="trend",
            description="12-1 industry momentum using equal-weighted same-industry returns from 252 trading days ago to 21 days ago.",
            params={"kind": "industry_momentum_skip", "window": 252, "skip_window": 21, "source_column": "industry_return_1_level_1"},
            tags=["industry", "trend", "momentum"],
            stability_level="candidate",
        ),
        _factor(
            name="price_to_26w_high",
            family="trend",
            description="Distance from the rolling 126-day closing high.",
            params={"kind": "price_to_rolling_high", "window": 126},
            tags=["price_only", "trend", "breakout"],
            stability_level="candidate",
        ),
        _factor(
            name="price_to_52w_high",
            family="trend",
            description="Distance from the rolling 252-day closing high.",
            params={"kind": "price_to_rolling_high", "window": 252},
            tags=["price_only", "trend", "breakout"],
            stability_level="candidate",
        ),
        *[
            _factor(
                name=f"short_term_reversal_{window}",
                family="trend",
                description=f"Negative of the recent {window}-day return to capture short-term reversal pressure.",
                params={"kind": "short_term_reversal", "window": window},
                tags=["price_only", "reversion"],
                stability_level="candidate",
            )
            for window in [5, 20]
        ],
        *[
            _factor(
                name=f"channel_position_{window}",
                family="position",
                description=f"Close location inside the {window}-day high-low channel.",
                params={"kind": "channel_position", "window": window},
                tags=["range", "position"],
                stability_level="stable" if window in {20} else "candidate",
            )
            for window in [10, 20, 60]
        ],
        *[
            _factor(
                name=f"distance_to_high_{window}",
                family="position",
                description=f"Distance from close to the {window}-day rolling high.",
                params={"kind": "distance_to_high", "window": window},
                tags=["range", "breakout"],
                stability_level="candidate",
            )
            for window in [10, 20, 60]
        ],
        *[
            _factor(
                name=f"distance_to_low_{window}",
                family="position",
                description=f"Distance from close to the {window}-day rolling low.",
                params={"kind": "distance_to_low", "window": window},
                tags=["range", "reversion"],
                stability_level="candidate",
            )
            for window in [10, 20, 60]
        ],
        _factor(
            name="rsi_14",
            family="oscillator",
            description="14-day relative strength index.",
            params={"kind": "rsi", "window": 14},
            tags=["oscillator"],
        ),
        _factor(
            name="bollinger_zscore_20",
            family="oscillator",
            description="20-day Bollinger z-score of close price.",
            params={"kind": "bollinger_zscore", "window": 20},
            tags=["oscillator", "mean_reversion"],
        ),
        _factor(
            name="stochastic_k_14",
            family="oscillator",
            description="14-day stochastic oscillator %K.",
            params={"kind": "stochastic_k", "window": 14},
            tags=["oscillator", "range"],
            stability_level="candidate",
        ),
        _factor(
            name="stochastic_d_14_3",
            family="oscillator",
            description="3-day smoothed stochastic oscillator %D based on 14-day %K.",
            params={"kind": "stochastic_d", "window": 14, "d_window": 3},
            tags=["oscillator", "range"],
            stability_level="candidate",
        ),
        _factor(
            name="williams_r_14",
            family="oscillator",
            description="14-day Williams %R oscillator.",
            params={"kind": "williams_r", "window": 14},
            tags=["oscillator", "range"],
            stability_level="candidate",
        ),
        _factor(
            name="money_flow_index_14",
            family="oscillator",
            description="14-day Money Flow Index using typical price and volume.",
            params={"kind": "money_flow_index", "window": 14},
            tags=["oscillator", "volume"],
            stability_level="candidate",
        ),
        _factor(
            name="intraday_range_pct",
            family="bar_shape",
            description="Daily high-low range scaled by close.",
            params={"kind": "intraday_range_pct"},
            tags=["bar_shape"],
        ),
        _factor(
            name="open_to_close_pct",
            family="bar_shape",
            description="Open-to-close move scaled by open.",
            params={"kind": "open_to_close_pct"},
            tags=["bar_shape"],
            stability_level="candidate",
        ),
        _factor(
            name="overnight_gap_pct",
            family="bar_shape",
            description="Gap from previous close to current open.",
            params={"kind": "overnight_gap_pct"},
            tags=["bar_shape", "gap"],
        ),
        _factor(
            name="close_location_value",
            family="bar_shape",
            description="Close location inside the day's high-low range.",
            params={"kind": "close_location_value"},
            tags=["bar_shape"],
        ),
        _factor(
            name="high_to_close_pct",
            family="bar_shape",
            description="Distance from high to close scaled by close.",
            params={"kind": "high_to_close_pct"},
            tags=["bar_shape"],
            stability_level="candidate",
        ),
        _factor(
            name="low_to_close_pct",
            family="bar_shape",
            description="Distance from low to close scaled by close.",
            params={"kind": "low_to_close_pct"},
            tags=["bar_shape"],
            stability_level="candidate",
        ),
        _factor(
            name="upper_shadow_pct",
            family="bar_shape",
            description="Upper shadow length scaled by close.",
            params={"kind": "upper_shadow_pct"},
            tags=["bar_shape", "candlestick"],
            stability_level="candidate",
        ),
        _factor(
            name="lower_shadow_pct",
            family="bar_shape",
            description="Lower shadow length scaled by close.",
            params={"kind": "lower_shadow_pct"},
            tags=["bar_shape", "candlestick"],
            stability_level="candidate",
        ),
        _factor(
            name="real_body_pct",
            family="bar_shape",
            description="Absolute candlestick body length scaled by open.",
            params={"kind": "real_body_pct"},
            tags=["bar_shape", "candlestick"],
            stability_level="candidate",
        ),
        *[
            _factor(
                name=f"beta_{window}_hs300",
                family="market",
                description=f"Rolling {window}-day beta versus HS300 benchmark returns.",
                params={"kind": "beta", "window": window, "benchmark_symbol": "sh000300"},
                tags=["benchmark", "risk"],
                stability_level="candidate",
            )
            for window in [20, 60, 120, 252]
        ],
        *[
            _factor(
                name=f"ivol_{window}_hs300",
                family="market",
                description=f"Rolling {window}-day residual volatility versus HS300 benchmark returns.",
                params={"kind": "idiosyncratic_volatility", "window": window, "benchmark_symbol": "sh000300"},
                tags=["benchmark", "risk"],
                stability_level="candidate",
            )
            for window in [20, 60, 120]
        ],
        _factor(
            name="amihud_illiquidity_5",
            family="liquidity",
            description="5-day average Amihud illiquidity using return divided by traded amount proxy.",
            params={"kind": "amihud", "window": 5},
            tags=["liquidity"],
            stability_level="candidate",
        ),
        _factor(
            name="amihud_illiquidity_20",
            family="liquidity",
            description="20-day average Amihud illiquidity using return divided by traded amount proxy.",
            params={"kind": "amihud", "window": 20},
            tags=["liquidity"],
            stability_level="candidate",
        ),
        _factor(
            name="amihud_illiquidity_60",
            family="liquidity",
            description="60-day average Amihud illiquidity using return divided by traded amount proxy.",
            params={"kind": "amihud", "window": 60},
            tags=["liquidity"],
            stability_level="candidate",
        ),
        *[
            _factor(
                name=f"turnover_{window}",
                family="liquidity",
                description=f"{window}-day rolling mean turnover ratio using traded volume and share capital.",
                params={"kind": "turnover", "window": window},
                tags=["liquidity", "turnover"],
                stability_level="candidate",
            )
            for window in [5, 20, 60]
        ],
        *[
            _factor(
                name=f"abnormal_turnover_{window}",
                family="liquidity",
                description=f"Current turnover relative to its {window}-day rolling mean.",
                params={"kind": "abnormal_turnover", "window": window},
                tags=["liquidity", "turnover"],
                stability_level="candidate",
            )
            for window in [5, 20, 60]
        ],
        *[
            _factor(
                name=f"dollar_volume_{window}",
                family="liquidity",
                description=f"{window}-day rolling mean traded amount proxy.",
                params={"kind": "dollar_volume", "window": window},
                tags=["liquidity", "volume"],
                stability_level="candidate",
            )
            for window in [5, 20, 60]
        ],
        _factor(
            name="log_total_mkt_cap",
            family="valuation",
            description="Log total market capitalization using close price and share capital.",
            params={"kind": "log_total_mkt_cap"},
            tags=["valuation", "size"],
            stability_level="candidate",
        ),
        _factor(
            name="book_to_market",
            family="valuation",
            description="Book-to-market ratio using parent equity per share divided by price.",
            params={"kind": "book_to_market"},
            tags=["valuation", "value"],
            stability_level="candidate",
        ),
        _factor(
            name="earnings_to_price",
            family="valuation",
            description="Earnings-to-price ratio using trailing parent net profit per share.",
            params={"kind": "earnings_to_price"},
            tags=["valuation", "value"],
            stability_level="candidate",
        ),
        _factor(
            name="sales_to_price",
            family="valuation",
            description="Sales-to-price ratio using trailing revenue per share.",
            params={"kind": "sales_to_price"},
            tags=["valuation", "value"],
            stability_level="candidate",
        ),
        _factor(
            name="cashflow_to_price",
            family="valuation",
            description="Operating-cashflow-to-price ratio using trailing operating cash flow per share.",
            params={"kind": "cashflow_to_price"},
            tags=["valuation", "value", "cashflow"],
            stability_level="candidate",
        ),
        _factor(
            name="dividend_yield_ttm",
            family="valuation",
            description="Trailing-12-month cash dividend yield using cash dividends per share divided by price.",
            params={"kind": "dividend_yield_ttm"},
            tags=["valuation", "yield", "dividend"],
            stability_level="candidate",
        ),
        _factor(
            name="gross_profit_to_assets",
            family="quality",
            description="Trailing gross profit scaled by total assets.",
            params={"kind": "gross_profit_to_assets"},
            tags=["quality", "profitability"],
            stability_level="candidate",
        ),
        _factor(
            name="operating_profitability",
            family="quality",
            description="Trailing operating profit scaled by parent equity.",
            params={"kind": "operating_profitability"},
            tags=["quality", "profitability"],
            stability_level="candidate",
        ),
        _factor(
            name="roe_ttm",
            family="quality",
            description="Trailing return on equity using parent net profit and parent equity.",
            params={"kind": "roe_ttm"},
            tags=["quality", "profitability"],
            stability_level="candidate",
        ),
        _factor(
            name="roa_ttm",
            family="quality",
            description="Trailing return on assets using parent net profit and total assets.",
            params={"kind": "roa_ttm"},
            tags=["quality", "profitability"],
            stability_level="candidate",
        ),
        _factor(
            name="gross_margin",
            family="quality",
            description="Trailing gross margin using revenue minus operating cost divided by revenue.",
            params={"kind": "gross_margin"},
            tags=["quality", "profitability", "margin"],
            stability_level="candidate",
        ),
        _factor(
            name="operating_margin",
            family="quality",
            description="Trailing operating profit divided by trailing revenue.",
            params={"kind": "operating_margin"},
            tags=["quality", "profitability", "margin"],
            stability_level="candidate",
        ),
        _factor(
            name="cash_profitability",
            family="quality",
            description="Trailing operating cash flow scaled by total assets.",
            params={"kind": "cash_profitability"},
            tags=["quality", "cashflow", "profitability"],
            stability_level="candidate",
        ),
        _factor(
            name="asset_turnover",
            family="quality",
            description="Trailing revenue scaled by total assets.",
            params={"kind": "asset_turnover"},
            tags=["quality", "efficiency"],
            stability_level="candidate",
        ),
        _factor(
            name="dividend_payout_ratio_ttm",
            family="quality",
            description="Trailing cash dividend per share divided by trailing earnings per share.",
            params={"kind": "dividend_payout_ratio_ttm"},
            tags=["quality", "payout", "dividend"],
            stability_level="candidate",
        ),
        _factor(
            name="liabilities_to_assets",
            family="quality",
            description="Total liabilities scaled by total assets as a simple balance-sheet leverage proxy.",
            params={"kind": "liabilities_to_assets"},
            tags=["quality", "safety", "leverage"],
            stability_level="candidate",
        ),
        _factor(
            name="liabilities_to_equity",
            family="quality",
            description="Total liabilities scaled by parent equity as a simple book leverage proxy.",
            params={"kind": "liabilities_to_equity"},
            tags=["quality", "safety", "leverage"],
            stability_level="candidate",
        ),
        _factor(
            name="cash_to_assets",
            family="quality",
            description="Monetary funds scaled by total assets as a balance-sheet safety proxy.",
            params={"kind": "cash_to_assets"},
            tags=["quality", "safety", "liquidity"],
            stability_level="candidate",
        ),
        _factor(
            name="inventory_to_assets",
            family="quality",
            description="Inventory scaled by total assets.",
            params={"kind": "inventory_to_assets"},
            tags=["quality", "working_capital"],
            stability_level="candidate",
        ),
        _factor(
            name="receivables_to_assets",
            family="quality",
            description="Accounts receivable scaled by total assets.",
            params={"kind": "receivables_to_assets"},
            tags=["quality", "working_capital"],
            stability_level="candidate",
        ),
        _factor(
            name="asset_growth",
            family="investment",
            description="Year-over-year growth in total assets using report-date fundamentals.",
            params={"kind": "asset_growth"},
            tags=["investment"],
            stability_level="candidate",
        ),
        _factor(
            name="investment_to_assets",
            family="investment",
            description="Trailing capital expenditure intensity scaled by lagged assets.",
            params={"kind": "investment_to_assets"},
            tags=["investment", "capex"],
            stability_level="candidate",
        ),
        _factor(
            name="inventory_growth",
            family="investment",
            description="Year-over-year inventory growth using report-date fundamentals.",
            params={"kind": "inventory_growth"},
            tags=["investment", "working_capital"],
            stability_level="candidate",
        ),
        _factor(
            name="receivables_growth",
            family="investment",
            description="Year-over-year receivables growth using report-date fundamentals.",
            params={"kind": "receivables_growth"},
            tags=["investment", "working_capital"],
            stability_level="candidate",
        ),
        _factor(
            name="capex_growth",
            family="investment",
            description="Year-over-year growth in trailing capital expenditure intensity level.",
            params={"kind": "capex_growth"},
            tags=["investment", "capex"],
            stability_level="candidate",
        ),
        _factor(
            name="accruals",
            family="quality",
            description="Trailing earnings minus operating cash flow scaled by lagged assets.",
            params={"kind": "accruals"},
            tags=["quality", "earnings_quality"],
            stability_level="candidate",
        ),
        *[
            _factor(
                name=f"downside_volatility_{window}",
                family="market",
                description=f"{window}-day downside semideviation of daily returns.",
                params={"kind": "downside_volatility", "window": window},
                tags=["risk", "downside"],
                stability_level="candidate",
            )
            for window in [20, 60]
        ],
        *[
            _factor(
                name=f"atr_{window}_pct",
                family="volatility",
                description=f"{window}-day average true range scaled by close.",
                params={"kind": "atr_pct", "window": window},
                tags=["risk", "range"],
                stability_level="candidate",
            )
            for window in [14, 20]
        ],
        *[
            _factor(
                name=f"parkinson_volatility_{window}",
                family="volatility",
                description=f"{window}-day Parkinson high-low volatility estimator.",
                params={"kind": "parkinson_volatility", "window": window},
                tags=["risk", "range"],
                stability_level="candidate",
            )
            for window in [20, 60]
        ],
        *[
            _factor(
                name=f"garman_klass_volatility_{window}",
                family="volatility",
                description=f"{window}-day Garman-Klass OHLC volatility estimator.",
                params={"kind": "garman_klass_volatility", "window": window},
                tags=["risk", "ohlc"],
                stability_level="candidate",
            )
            for window in [20, 60]
        ],
    ]
)


def get_default_registry() -> FactorRegistry:
    return DEFAULT_FACTOR_REGISTRY
