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
        for window in [1, 3, 5, 10, 20, 60]
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
        for window in [5, 10, 20, 60]
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
        for window in [5, 10, 20]
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
        for window in [5, 20]
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
            for window in [10, 20]
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
            for window in [10, 20]
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
            for window in [10, 20]
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
        *[
            _factor(
                name=f"beta_{window}_hs300",
                family="market",
                description=f"Rolling {window}-day beta versus HS300 benchmark returns.",
                params={"kind": "beta", "window": window, "benchmark_symbol": "sh000300"},
                tags=["benchmark", "risk"],
                stability_level="candidate",
            )
            for window in [60, 120]
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
            for window in [20, 60]
        ],
        _factor(
            name="amihud_illiquidity_20",
            family="liquidity",
            description="20-day average Amihud illiquidity using return divided by traded amount proxy.",
            params={"kind": "amihud", "window": 20},
            tags=["liquidity"],
            stability_level="candidate",
        ),
        _factor(
            name="turnover_20",
            family="liquidity",
            description="20-day rolling mean turnover ratio using traded volume and share capital.",
            params={"kind": "turnover", "window": 20},
            tags=["liquidity", "turnover"],
            stability_level="candidate",
        ),
        _factor(
            name="abnormal_turnover_20",
            family="liquidity",
            description="Current turnover relative to its 20-day rolling mean.",
            params={"kind": "abnormal_turnover", "window": 20},
            tags=["liquidity", "turnover"],
            stability_level="candidate",
        ),
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
            name="accruals",
            family="quality",
            description="Trailing earnings minus operating cash flow scaled by lagged assets.",
            params={"kind": "accruals"},
            tags=["quality", "earnings_quality"],
            stability_level="candidate",
        ),
    ]
)


def get_default_registry() -> FactorRegistry:
    return DEFAULT_FACTOR_REGISTRY
