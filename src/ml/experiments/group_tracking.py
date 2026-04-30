from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


GROUP_TRACKING_SCHEMA_VERSION = 1


class GroupRunTracker:
    def __init__(
        self,
        *,
        group_name: str,
        experiment_paths: list[str],
        output_dir: Path,
        mode: str,
        continue_on_error: bool,
        resume: bool,
        execution_options: dict[str, object] | None = None,
    ) -> None:
        self.group_name = str(group_name)
        self.experiment_paths = [str(path) for path in experiment_paths]
        self.output_dir = Path(output_dir)
        self.mode = str(mode)
        self.continue_on_error = bool(continue_on_error)
        self.resume = bool(resume)
        self.execution_options = dict(execution_options or {})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "group_summary.json"
        self.manifest_path = self.output_dir / f"group_{self.mode}_manifest.json"
        self.failures_path = self.output_dir / f"group_{self.mode}_failures.json"
        self.started_at = _utc_now()
        self.updated_at = self.started_at
        self.finished_at: str | None = None
        self.status = "running"
        self.experiments_by_path: dict[str, dict[str, object]] = {}
        self.failures_by_path: dict[str, dict[str, object]] = {}
        self.resume_skipped_paths: set[str] = set()
        self.tasks: dict[str, dict[str, object]] = {
            path: {
                "experiment_path": path,
                "status": "pending",
                "resume_skipped": False,
                "updated_at": self.started_at,
            }
            for path in self.experiment_paths
        }
        if self.resume:
            self._load_existing_completions()
        self.write()

    def pending_paths(self) -> list[str]:
        return [
            path
            for path in self.experiment_paths
            if path not in self.experiments_by_path
        ]

    @property
    def failed_count(self) -> int:
        return len(self.failures_by_path)

    def record_completed(self, summary: dict[str, object], *, resume_skipped: bool = False) -> None:
        experiment_path = str(summary.get("experiment_path") or "")
        if not experiment_path:
            raise ValueError("Experiment summary is missing experiment_path.")
        self.updated_at = _utc_now()
        self.experiments_by_path[experiment_path] = dict(summary)
        if experiment_path in self.failures_by_path:
            self.failures_by_path.pop(experiment_path, None)
        task = self.tasks.setdefault(
            experiment_path,
            {"experiment_path": experiment_path},
        )
        task.update(
            {
                "status": "completed",
                "name": summary.get("name"),
                "report_dir": summary.get("report_dir"),
                "artifact_path": summary.get("artifact_path"),
                "resume_skipped": bool(resume_skipped),
                "updated_at": self.updated_at,
                "error": None,
            }
        )
        if resume_skipped:
            self.resume_skipped_paths.add(experiment_path)
        else:
            self.resume_skipped_paths.discard(experiment_path)
        self.write()

    def record_failure(self, *, experiment_path: str, error: str) -> None:
        normalized_path = str(experiment_path)
        self.updated_at = _utc_now()
        failure = {
            "experiment_path": normalized_path,
            "error": str(error),
            "mode": self.mode,
            "failed_at": self.updated_at,
        }
        self.failures_by_path[normalized_path] = failure
        task = self.tasks.setdefault(
            normalized_path,
            {"experiment_path": normalized_path},
        )
        task.update(
            {
                "status": "failed",
                "resume_skipped": False,
                "updated_at": self.updated_at,
                "error": str(error),
            }
        )
        self.experiments_by_path.pop(normalized_path, None)
        self.resume_skipped_paths.discard(normalized_path)
        self.write()

    def finalize(self, *, status: str) -> dict[str, object]:
        self.status = str(status)
        self.finished_at = _utc_now()
        self.updated_at = self.finished_at
        return self.write()

    def write(self) -> dict[str, object]:
        manifest = self._build_manifest_payload()
        summary = self._build_summary_payload()
        failures = self._build_failures_payload()
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _load_existing_completions(self) -> None:
        summary_payload = None
        if self.summary_path.exists():
            summary_payload = json.loads(self.summary_path.read_text(encoding="utf-8"))

        completed_paths: set[str] = set()
        if self.manifest_path.exists():
            manifest_payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if str(manifest_payload.get("mode") or "") == self.mode:
                completed_paths = {
                    str(task.get("experiment_path"))
                    for task in list(manifest_payload.get("tasks", []))
                    if str(task.get("status")) == "completed" and str(task.get("experiment_path") or "") in self.tasks
                }
        elif summary_payload is not None:
            summary_mode = summary_payload.get("mode")
            if summary_mode in {None, self.mode}:
                completed_paths = {
                    str(item.get("experiment_path"))
                    for item in list(summary_payload.get("experiments", []))
                    if str(item.get("experiment_path") or "") in self.tasks
                }

        if summary_payload is not None:
            summaries_by_path = {
                str(item.get("experiment_path")): dict(item)
                for item in list(summary_payload.get("experiments", []))
                if str(item.get("experiment_path") or "")
            }
            for experiment_path in completed_paths:
                item = summaries_by_path.get(experiment_path)
                if item is None:
                    continue
                self.record_completed(item, resume_skipped=True)
        for path in self.pending_paths():
            task = self.tasks[path]
            task.update({"status": "pending", "resume_skipped": False, "error": None, "updated_at": self.started_at})

    def _build_manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": GROUP_TRACKING_SCHEMA_VERSION,
            "name": self.group_name,
            "mode": self.mode,
            "status": self.status,
            "continue_on_error": self.continue_on_error,
            "resume": self.resume,
            "execution_options": self.execution_options,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "experiment_paths": list(self.experiment_paths),
            "tasks": [self.tasks[path] for path in self.experiment_paths],
        }

    def _build_summary_payload(self) -> dict[str, object]:
        experiments = [self.experiments_by_path[path] for path in self.experiment_paths if path in self.experiments_by_path]
        failures = [self.failures_by_path[path] for path in self.experiment_paths if path in self.failures_by_path]
        completed_count = len(experiments)
        failed_count = len(failures)
        skipped_count = len(self.resume_skipped_paths)
        pending_count = len(self.experiment_paths) - completed_count - failed_count
        return {
            "schema_version": GROUP_TRACKING_SCHEMA_VERSION,
            "name": self.group_name,
            "mode": self.mode,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "experiment_paths": list(self.experiment_paths),
            "experiment_count": len(self.experiment_paths),
            "completed_count": completed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "pending_count": pending_count,
            "manifest_path": str(self.manifest_path),
            "failures_path": str(self.failures_path),
            "experiments": experiments,
            "failures": failures,
        }

    def _build_failures_payload(self) -> dict[str, object]:
        failures = [self.failures_by_path[path] for path in self.experiment_paths if path in self.failures_by_path]
        return {
            "schema_version": GROUP_TRACKING_SCHEMA_VERSION,
            "name": self.group_name,
            "mode": self.mode,
            "status": self.status,
            "updated_at": self.updated_at,
            "failure_count": len(failures),
            "failures": failures,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
