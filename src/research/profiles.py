from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_RESEARCH_PROFILE_ROOT = Path("configs") / "research_profiles"


@dataclass(slots=True, frozen=True)
class ResearchProfileSpec:
    name: str
    market: str = "ashare"
    provider: str = "parquet"
    data_root: str = "data/lake"
    universe_root: str = "data/universe"
    reference_root: str = "data/reference"
    timeframe: str = "1d"
    adjust: str = "qfq"
    universe: str | None = None
    benchmark_symbol: str = "sh000300"
    industry_standard: str = "申银万国行业分类标准"
    market_cap_bucket_count: int = 5
    official_baseline_manifest: str | None = None


def resolve_research_profile_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    stem = str(profile)
    search_paths = [
        DEFAULT_RESEARCH_PROFILE_ROOT / f"{stem}.yaml",
        DEFAULT_RESEARCH_PROFILE_ROOT / f"{stem}.yml",
        DEFAULT_RESEARCH_PROFILE_ROOT / "ashare" / f"{stem}.yaml",
        DEFAULT_RESEARCH_PROFILE_ROOT / "ashare" / f"{stem}.yml",
    ]
    for path in search_paths:
        if path.exists():
            return path
    searched = ", ".join(str(item) for item in search_paths)
    raise FileNotFoundError(f"Research profile was not found: {profile}. Tried: {searched}")


def load_research_profile(profile: str | Path) -> ResearchProfileSpec:
    profile_path = resolve_research_profile_path(profile)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    return ResearchProfileSpec(
        name=str(raw.get("name") or profile_path.stem),
        market=str(raw.get("market", "ashare")),
        provider=str(raw.get("provider", "parquet")),
        data_root=str(raw.get("data_root", "data/lake")),
        universe_root=str(raw.get("universe_root", "data/universe")),
        reference_root=str(raw.get("reference_root", "data/reference")),
        timeframe=str(raw.get("timeframe", "1d")),
        adjust=str(raw.get("adjust", "qfq")),
        universe=str(raw["universe"]) if raw.get("universe") is not None else None,
        benchmark_symbol=str(raw.get("benchmark_symbol", "sh000300")),
        industry_standard=str(raw.get("industry_standard", "申银万国行业分类标准")),
        market_cap_bucket_count=int(raw.get("market_cap_bucket_count", 5)),
        official_baseline_manifest=(
            str(raw["official_baseline_manifest"])
            if raw.get("official_baseline_manifest") is not None
            else None
        ),
    )


def apply_profile_defaults(
    raw: dict[str, object],
    *,
    profile: ResearchProfileSpec | None,
) -> dict[str, object]:
    merged = dict(raw)
    if profile is None:
        return merged
    defaults: dict[str, object] = {
        "research_profile": profile.name,
        "market": profile.market,
        "provider": profile.provider,
        "data_root": profile.data_root,
        "universe_root": profile.universe_root,
        "reference_root": profile.reference_root,
        "timeframe": profile.timeframe,
        "adjust": profile.adjust,
        "universe": profile.universe,
        "benchmark_symbol": profile.benchmark_symbol,
        "industry_standard": profile.industry_standard,
        "market_cap_bucket_count": profile.market_cap_bucket_count,
        "baseline_manifest_path": profile.official_baseline_manifest,
    }
    for key, value in defaults.items():
        if key not in merged or merged.get(key) in {None, ""}:
            merged[key] = value
    return merged
