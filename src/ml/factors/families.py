from __future__ import annotations

from ml.factors.specs import FactorFamilySpec


FACTOR_FAMILIES = {
    "trend": FactorFamilySpec(
        family_name="trend",
        purpose="Describe medium-term direction and extension.",
        max_recommended_count=3,
        notes="Prefer one or two anchor windows before adding extra variants.",
    ),
    "volatility": FactorFamilySpec(
        family_name="volatility",
        purpose="Describe recent instability and return dispersion.",
        max_recommended_count=2,
        notes="Too many nearby windows often add little new information.",
    ),
    "volume": FactorFamilySpec(
        family_name="volume",
        purpose="Describe participation and turnover acceleration.",
        max_recommended_count=3,
        notes="Mix shock-style and trend-style volume factors before adding more windows.",
    ),
    "position": FactorFamilySpec(
        family_name="position",
        purpose="Describe price location inside recent ranges.",
        max_recommended_count=2,
        notes="Channel position and edge distance factors overlap heavily.",
    ),
    "oscillator": FactorFamilySpec(
        family_name="oscillator",
        purpose="Describe stretch, reversion pressure, and normalized deviation.",
        max_recommended_count=2,
        notes="Usually one directional oscillator plus one normalized distance is enough.",
    ),
    "bar_shape": FactorFamilySpec(
        family_name="bar_shape",
        purpose="Describe daily structure, gaps, and close strength.",
        max_recommended_count=4,
        notes="These are micro-behavior factors and can be combined in small packs.",
    ),
    "market": FactorFamilySpec(
        family_name="market",
        purpose="Describe market-relative exposure and residual risk.",
        max_recommended_count=3,
        notes="Keep benchmark-relative features compact to avoid over-weighting one reference index.",
    ),
    "liquidity": FactorFamilySpec(
        family_name="liquidity",
        purpose="Describe trading capacity, turnover, and price impact.",
        max_recommended_count=3,
        notes="Price-impact and turnover features complement each other better than multiple turnover windows.",
    ),
    "valuation": FactorFamilySpec(
        family_name="valuation",
        purpose="Describe price relative to book value, earnings, sales, and size.",
        max_recommended_count=4,
        notes="Valuation factors should come from point-in-time fundamentals aligned by publish date.",
    ),
    "quality": FactorFamilySpec(
        family_name="quality",
        purpose="Describe profitability, capital efficiency, and earnings quality.",
        max_recommended_count=3,
        notes="Quality factors often overlap, so keep only a few broad anchors per experiment.",
    ),
    "investment": FactorFamilySpec(
        family_name="investment",
        purpose="Describe balance-sheet expansion and capital spending intensity.",
        max_recommended_count=2,
        notes="Investment signals are slower-moving and should usually be paired with faster technical features.",
    ),
    "rolling_stats": FactorFamilySpec(
        family_name="rolling_stats",
        purpose="Describe rolling statistical transforms such as regression slope, correlation, and quantiles.",
        max_recommended_count=4,
        notes="Prefer a compact subset because many rolling statistics overlap heavily.",
    ),
    "raw_price_history": FactorFamilySpec(
        family_name="raw_price_history",
        purpose="Describe normalized lagged price history features compatible with raw-history model inputs.",
        max_recommended_count=10,
        notes="These are high-dimensional sequence-style features and are usually used as a pack.",
    ),
    "raw_volume_history": FactorFamilySpec(
        family_name="raw_volume_history",
        purpose="Describe normalized lagged volume history features compatible with raw-history model inputs.",
        max_recommended_count=10,
        notes="These are high-dimensional sequence-style features and are usually used with raw price history.",
    ),
}


def list_families() -> list[FactorFamilySpec]:
    return [FACTOR_FAMILIES[name] for name in sorted(FACTOR_FAMILIES)]


def get_family(name: str) -> FactorFamilySpec:
    try:
        return FACTOR_FAMILIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(FACTOR_FAMILIES))
        raise ValueError(f"Unknown factor family: {name}. Available families: {available}") from exc
