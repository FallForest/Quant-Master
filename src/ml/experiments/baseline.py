from __future__ import annotations

import json
from pathlib import Path


OFFICIAL_BASELINE_MANIFEST_PATH = Path("configs") / "experiments" / "official" / "hs300_official_baseline_manifest.json"
PROMOTION_METRIC_KEYS = (
    "full_oos_spearman_ic",
    "window_mean_spearman_ic",
    "window_min_spearman_ic",
    "window_mean_ic_ir",
)


def load_official_baseline_manifest(path: str | Path | None = None) -> dict[str, object] | None:
    manifest_path = Path(path) if path is not None else OFFICIAL_BASELINE_MANIFEST_PATH
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def evaluate_promotion_gate(
    candidate: dict[str, object],
    *,
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    if manifest is None:
        return {
            "official_baseline_name": None,
            "is_official_baseline": False,
            "beats_official_baseline": None,
            "baseline_gate_failures": [],
        }
    baseline_name = str(manifest.get("baseline_name", ""))
    baseline_metrics = dict(manifest.get("promotion_rule", {})).get("metrics", {})
    failures: list[str] = []
    result: dict[str, object] = {
        "official_baseline_name": baseline_name,
        "is_official_baseline": str(candidate.get("name", "")) == baseline_name,
    }
    for metric_key in PROMOTION_METRIC_KEYS:
        baseline_value = float(dict(baseline_metrics).get(metric_key, 0.0))
        candidate_value = float(candidate.get(metric_key, 0.0))
        passed = candidate_value > baseline_value
        result[f"beats_{metric_key}"] = passed
        if not passed:
            failures.append(metric_key)
    result["beats_official_baseline"] = False if result["is_official_baseline"] else len(failures) == 0
    result["baseline_gate_failures"] = failures
    return result
