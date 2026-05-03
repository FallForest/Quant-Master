from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.experiments.loader import load_experiment_group_spec


def load_group_summary(group: str | Path) -> dict[str, object]:
    group_path = Path(group)
    if group_path.exists():
        if group_path.suffix.lower() == ".json":
            return json.loads(group_path.read_text(encoding="utf-8"))
        if group_path.suffix.lower() in {".yaml", ".yml"}:
            spec = load_experiment_group_spec(group_path)
            default_path = Path(spec.output_dir or Path("reports") / "experiments" / "groups" / spec.name) / "group_summary.json"
            if not default_path.exists():
                raise FileNotFoundError(f"Group summary was not found: {default_path}")
            return json.loads(default_path.read_text(encoding="utf-8"))
    default_summary = Path("reports") / "experiments" / "groups" / str(group) / "group_summary.json"
    if default_summary.exists():
        return json.loads(default_summary.read_text(encoding="utf-8"))
    for candidate in Path(".").rglob("group_summary.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if payload.get("name") == str(group):
            return payload
    raise FileNotFoundError(f"Could not resolve experiment group summary for: {group}")


def build_comparison_frame(group_summary: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in group_summary.get("experiments", []):
        training_metadata = dict(item.get("training_metadata", {}))
        tuning_metadata = dict(training_metadata.get("tuning", {}))
        validation_metrics = dict(item.get("validation_metrics", {}))
        signal_test_metrics = dict(item.get("signal_test_metrics", {}))
        diagnostics = dict(item.get("research_diagnostics", {}))
        candidate_selection = dict(item.get("candidate_selection", {}))
        row = {
            "name": item.get("name"),
            "group": item.get("group"),
            "research_profile": item.get("research_profile") or training_metadata.get("research_profile"),
            "model": item.get("model"),
            "feature_normalization": item.get("feature_normalization", training_metadata.get("feature_normalization")),
            "feature_count": len(item.get("features", []) or training_metadata.get("feature_columns", [])),
            "validation_mode": tuning_metadata.get("validation_mode") or training_metadata.get("validation_mode"),
            "selection_validation_mode": training_metadata.get("selection_validation_mode")
            or training_metadata.get("validation_mode"),
            "target_mode": training_metadata.get("target_mode"),
            "tuned": bool(tuning_metadata.get("enabled")),
            "ic": validation_metrics.get("ic", 0.0),
            "ic_std": validation_metrics.get("ic_std", 0.0),
            "icir": validation_metrics.get("icir", 0.0),
            "rank_ic": validation_metrics.get("rank_ic", 0.0),
            "rank_ic_std": validation_metrics.get("rank_ic_std", 0.0),
            "rank_icir": validation_metrics.get("rank_icir", 0.0),
            "test_ic": signal_test_metrics.get("ic", 0.0),
            "test_ic_std": signal_test_metrics.get("ic_std", 0.0),
            "test_icir": signal_test_metrics.get("icir", 0.0),
            "test_rank_ic": signal_test_metrics.get("rank_ic", 0.0),
            "test_rank_ic_std": signal_test_metrics.get("rank_ic_std", 0.0),
            "test_rank_icir": signal_test_metrics.get("rank_icir", 0.0),
            "mae": validation_metrics.get("mae", 0.0),
            "r2": validation_metrics.get("r2", 0.0),
            "pbo": diagnostics.get("pbo", 0.0),
            "dsr": diagnostics.get("dsr", 0.0),
            "candidate_metric": candidate_selection.get("selected_metric"),
            "report_dir": item.get("report_dir"),
            "artifact_path": item.get("artifact_path"),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["test_rank_ic", "test_rank_icir", "rank_ic", "ic"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def render_comparison_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No experiment results found."
    view = frame.copy()
    for column in [
        "ic",
        "ic_std",
        "icir",
        "rank_ic",
        "rank_ic_std",
        "rank_icir",
        "test_ic",
        "test_ic_std",
        "test_icir",
        "test_rank_ic",
        "test_rank_ic_std",
        "test_rank_icir",
        "ndcg_at_10",
        "mae",
        "r2",
        "pbo",
        "dsr",
    ]:
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return view.to_string(index=False)
