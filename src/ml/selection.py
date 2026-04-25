from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CandidateSelectionConfig:
    top_k: int = 5
    metric: str = "oos_spearman_ic"
    direction: str = "maximize"

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k": int(self.top_k),
            "metric": self.metric,
            "direction": self.direction,
        }


def score_candidate_metrics(metrics: dict[str, float], *, metric: str, direction: str) -> float:
    value = float(metrics.get(metric, 0.0))
    return value if direction != "minimize" else -value
