from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FactorSpec:
    name: str
    family: str
    version: str
    description: str
    input_columns: list[str]
    params: dict[str, object] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    stability_level: str = "stable"


@dataclass(slots=True)
class FactorFamilySpec:
    family_name: str
    purpose: str
    max_recommended_count: int
    notes: str = ""
