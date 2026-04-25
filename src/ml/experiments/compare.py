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
        oos_pearson_ic = validation_metrics.get("oos_pearson_ic", validation_metrics.get("pearson_ic", 0.0))
        oos_spearman_ic = validation_metrics.get("oos_spearman_ic", validation_metrics.get("spearman_ic", 0.0))
        oos_ic_std = validation_metrics.get("oos_ic_std", validation_metrics.get("ic_std", 0.0))
        oos_ic_ir = validation_metrics.get("oos_ic_ir", validation_metrics.get("ic_ir", 0.0))
        oos_ndcg_at_10 = validation_metrics.get("oos_ndcg_at_10", validation_metrics.get("ndcg_at_10", 0.0))
        test_pearson_ic = signal_test_metrics.get("oos_pearson_ic", signal_test_metrics.get("pearson_ic", 0.0))
        test_spearman_ic = signal_test_metrics.get("oos_spearman_ic", signal_test_metrics.get("spearman_ic", 0.0))
        test_ic_std = signal_test_metrics.get("oos_ic_std", signal_test_metrics.get("ic_std", 0.0))
        test_ic_ir = signal_test_metrics.get("oos_ic_ir", signal_test_metrics.get("ic_ir", 0.0))
        test_ndcg_at_10 = signal_test_metrics.get("oos_ndcg_at_10", signal_test_metrics.get("ndcg_at_10", 0.0))
        rows.append(
            {
                "name": item.get("name"),
                "group": item.get("group"),
                "model": item.get("model"),
                "feature_count": len(item.get("features", []) or training_metadata.get("feature_columns", [])),
                "validation_mode": tuning_metadata.get("validation_mode") or training_metadata.get("validation_mode"),
                "selection_validation_mode": training_metadata.get("selection_validation_mode")
                or training_metadata.get("validation_mode"),
                "target_mode": training_metadata.get("target_mode"),
                "tuned": bool(tuning_metadata.get("enabled")),
                "pearson_ic": oos_pearson_ic,
                "spearman_ic": oos_spearman_ic,
                "ic_std": oos_ic_std,
                "ic_ir": oos_ic_ir,
                "ndcg_at_10": oos_ndcg_at_10,
                "oos_pearson_ic": oos_pearson_ic,
                "oos_spearman_ic": oos_spearman_ic,
                "oos_ic_std": oos_ic_std,
                "oos_ic_ir": oos_ic_ir,
                "oos_ndcg_at_10": oos_ndcg_at_10,
                "test_pearson_ic": test_pearson_ic,
                "test_spearman_ic": test_spearman_ic,
                "test_ic_std": test_ic_std,
                "test_ic_ir": test_ic_ir,
                "test_ndcg_at_10": test_ndcg_at_10,
                "signal_test_rows": item.get("signal_test_rows", 0),
                "mae": validation_metrics.get("mae", 0.0),
                "r2": validation_metrics.get("r2", 0.0),
                "pbo": diagnostics.get("pbo", 0.0),
                "dsr": diagnostics.get("dsr", 0.0),
                "candidate_metric": candidate_selection.get("selected_metric"),
                "report_dir": item.get("report_dir"),
                "artifact_path": item.get("artifact_path"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["test_spearman_ic", "test_ic_ir", "oos_spearman_ic"], ascending=[False, False, False]).reset_index(drop=True)


def render_comparison_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No experiment results found."
    view = frame.copy()
    for column in [
        "pearson_ic",
        "spearman_ic",
        "ic_std",
        "ic_ir",
        "ndcg_at_10",
        "oos_pearson_ic",
        "oos_spearman_ic",
        "oos_ic_std",
        "oos_ic_ir",
        "oos_ndcg_at_10",
        "test_pearson_ic",
        "test_spearman_ic",
        "test_ic_std",
        "test_ic_ir",
        "test_ndcg_at_10",
        "mae",
        "r2",
        "pbo",
        "dsr",
    ]:
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return view.to_string(index=False)
