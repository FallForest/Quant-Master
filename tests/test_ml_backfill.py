from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.backfill import backfill_signal_metrics_group_from_path


def test_backfill_group_resume_skips_completed_experiments(tmp_path: Path, monkeypatch) -> None:
    group_file = tmp_path / "demo_group.yaml"
    group_output_dir = tmp_path / "reports" / "experiments" / "groups" / "demo_group"
    experiment_a = str((tmp_path / "a.yaml").resolve())
    experiment_b = str((tmp_path / "b.yaml").resolve())
    group_file.write_text(
        "\n".join(
            [
                "name: demo_group",
                "experiments:",
                "  - a.yaml",
                "  - b.yaml",
                f"output_dir: {group_output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    group_output_dir.mkdir(parents=True, exist_ok=True)
    (group_output_dir / "group_summary.json").write_text(
        json.dumps(
            {
                "name": "demo_group",
                "experiment_paths": [experiment_a, experiment_b],
                "experiment_count": 2,
                "completed_count": 1,
                "failed_count": 0,
                "experiments": [
                    {
                        "name": "exp_a",
                        "experiment_path": experiment_a,
                        "report_dir": str(tmp_path / "reports" / "a"),
                        "artifact_path": str(tmp_path / "artifacts" / "a"),
                    }
                ],
                "failures": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    visited: list[str] = []

    def fake_backfill(path, artifact_path_override=None, report_dir_override=None):
        visited.append(str(path))
        return {
            "name": "exp_b",
            "experiment_path": experiment_b,
            "report_dir": str(tmp_path / "reports" / "b"),
            "artifact_path": str(tmp_path / "artifacts" / "b"),
        }

    monkeypatch.setattr("ml.backfill.backfill_signal_metrics_from_path", fake_backfill)

    summary = backfill_signal_metrics_group_from_path(group_file, resume=True)

    assert visited == [experiment_b]
    assert summary["completed_count"] == 2
    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 0
