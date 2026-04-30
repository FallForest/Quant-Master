from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PIPELINE_TRACKING_SCHEMA_VERSION = 1


class PipelineRunTracker:
    def __init__(
        self,
        *,
        pipeline_name: str,
        output_dir: Path,
        item_label: str,
        items: list[str],
        options: dict[str, object] | None = None,
    ) -> None:
        self.pipeline_name = str(pipeline_name)
        self.output_dir = Path(output_dir)
        self.item_label = str(item_label)
        self.items = [str(item) for item in items]
        self.options = dict(options or {})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / f"{self.pipeline_name}_summary.json"
        self.manifest_path = self.output_dir / f"{self.pipeline_name}_manifest.json"
        self.failures_path = self.output_dir / f"{self.pipeline_name}_failures.json"
        self.started_at = _utc_now()
        self.updated_at = self.started_at
        self.finished_at: str | None = None
        self.status = "running"
        self.results_by_item: dict[str, dict[str, object]] = {}
        self.failures_by_item: dict[str, dict[str, object]] = {}
        self.tasks: dict[str, dict[str, object]] = {
            item: {
                self.item_label: item,
                "status": "pending",
                "updated_at": self.started_at,
                "error": None,
            }
            for item in self.items
        }
        self.write()

    def record_result(self, item: str, payload: dict[str, object]) -> None:
        normalized = str(item)
        self.updated_at = _utc_now()
        self.results_by_item[normalized] = dict(payload)
        self.failures_by_item.pop(normalized, None)
        task = self.tasks.setdefault(normalized, {self.item_label: normalized})
        task.update(
            {
                "status": "completed",
                "updated_at": self.updated_at,
                "error": None,
            }
        )
        for key in ("path", "industry_path", "dividend_path", "skipped", "attempts"):
            if key in payload:
                task[key] = payload[key]
        self.write()

    def record_failure(self, *, item: str, error: str, attempts: int | None = None) -> None:
        normalized = str(item)
        self.updated_at = _utc_now()
        failure = {
            self.item_label: normalized,
            "error": str(error),
            "failed_at": self.updated_at,
        }
        if attempts is not None:
            failure["attempts"] = int(attempts)
        self.failures_by_item[normalized] = failure
        self.results_by_item.pop(normalized, None)
        task = self.tasks.setdefault(normalized, {self.item_label: normalized})
        task.update(
            {
                "status": "failed",
                "updated_at": self.updated_at,
                "error": str(error),
            }
        )
        if attempts is not None:
            task["attempts"] = int(attempts)
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

    def _build_manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": PIPELINE_TRACKING_SCHEMA_VERSION,
            "pipeline_name": self.pipeline_name,
            "item_label": self.item_label,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "options": self.options,
            "items": list(self.items),
            "tasks": [self.tasks[item] for item in self.items if item in self.tasks],
        }

    def _build_summary_payload(self) -> dict[str, object]:
        completed = [self.results_by_item[item] for item in self.items if item in self.results_by_item]
        failures = [self.failures_by_item[item] for item in self.items if item in self.failures_by_item]
        completed_count = len(completed)
        failed_count = len(failures)
        pending_count = len(self.items) - completed_count - failed_count
        skipped_count = sum(1 for item in completed if bool(item.get("skipped")))
        return {
            "schema_version": PIPELINE_TRACKING_SCHEMA_VERSION,
            "pipeline_name": self.pipeline_name,
            "item_label": self.item_label,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "item_count": len(self.items),
            "completed_count": completed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "pending_count": pending_count,
            "manifest_path": str(self.manifest_path),
            "failures_path": str(self.failures_path),
            "results": completed,
            "failures": failures,
        }

    def _build_failures_payload(self) -> dict[str, object]:
        failures = [self.failures_by_item[item] for item in self.items if item in self.failures_by_item]
        return {
            "schema_version": PIPELINE_TRACKING_SCHEMA_VERSION,
            "pipeline_name": self.pipeline_name,
            "item_label": self.item_label,
            "status": self.status,
            "updated_at": self.updated_at,
            "failure_count": len(failures),
            "failures": failures,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
