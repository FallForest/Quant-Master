from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, log
from statistics import NormalDist

import pandas as pd


STANDARD_NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329


@dataclass(slots=True)
class OverfitDiagnostics:
    pbo: float | None
    combination_count: int
    trial_count: int
    fold_count: int
    selected_lambda: float | None
    selected_oos_rank_pct: float | None
    dsr: float | None
    observed_sharpe: float | None
    benchmark_sharpe: float | None
    return_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "pbo": self.pbo,
            "combination_count": self.combination_count,
            "trial_count": self.trial_count,
            "fold_count": self.fold_count,
            "selected_lambda": self.selected_lambda,
            "selected_oos_rank_pct": self.selected_oos_rank_pct,
            "dsr": self.dsr,
            "observed_sharpe": self.observed_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "return_count": self.return_count,
        }


def build_overfit_diagnostics(
    *,
    trial_records: list[dict[str, object]],
    direction: str,
    returns: pd.Series,
) -> OverfitDiagnostics:
    pbo = compute_pbo(trial_records, direction=direction)
    dsr = compute_deflated_sharpe_ratio(returns, trial_count=max(1, len(trial_records)))
    return OverfitDiagnostics(
        pbo=pbo["pbo"],
        combination_count=int(pbo["combination_count"]),
        trial_count=int(pbo["trial_count"]),
        fold_count=int(pbo["fold_count"]),
        selected_lambda=pbo["selected_lambda"],
        selected_oos_rank_pct=pbo["selected_oos_rank_pct"],
        dsr=dsr["dsr"],
        observed_sharpe=dsr["observed_sharpe"],
        benchmark_sharpe=dsr["benchmark_sharpe"],
        return_count=int(dsr["return_count"]),
    )


def compute_pbo(trial_records: list[dict[str, object]], *, direction: str = "maximize") -> dict[str, object]:
    score_matrix = _build_trial_fold_matrix(trial_records)
    if score_matrix is None:
        return {
            "pbo": None,
            "combination_count": 0,
            "trial_count": len(trial_records),
            "fold_count": 0,
            "selected_lambda": None,
            "selected_oos_rank_pct": None,
        }

    trial_count = len(score_matrix)
    fold_count = len(score_matrix[0])
    if trial_count < 2 or fold_count < 2:
        return {
            "pbo": None,
            "combination_count": 0,
            "trial_count": trial_count,
            "fold_count": fold_count,
            "selected_lambda": None,
            "selected_oos_rank_pct": None,
        }

    maximize = direction != "minimize"
    lambdas: list[float] = []
    oos_rank_pcts: list[float] = []
    train_fold_count = max(1, fold_count // 2)

    for train_indices in combinations(range(fold_count), train_fold_count):
        test_indices = tuple(index for index in range(fold_count) if index not in train_indices)
        if not test_indices:
            continue

        train_scores = [
            sum(scores[index] for index in train_indices) / len(train_indices)
            for scores in score_matrix
        ]
        selected_index = _best_index(train_scores, maximize=maximize)
        test_scores = [
            sum(scores[index] for index in test_indices) / len(test_indices)
            for scores in score_matrix
        ]
        rank_pct = _rank_percentile(test_scores, selected_index=selected_index, maximize=maximize)
        oos_rank_pcts.append(rank_pct)
        lambdas.append(_logit(rank_pct))

    if not lambdas:
        return {
            "pbo": None,
            "combination_count": 0,
            "trial_count": trial_count,
            "fold_count": fold_count,
            "selected_lambda": None,
            "selected_oos_rank_pct": None,
        }

    return {
        "pbo": float(sum(1 for value in lambdas if value <= 0.0) / len(lambdas)),
        "combination_count": len(lambdas),
        "trial_count": trial_count,
        "fold_count": fold_count,
        "selected_lambda": float(sum(lambdas) / len(lambdas)),
        "selected_oos_rank_pct": float(sum(oos_rank_pcts) / len(oos_rank_pcts)),
    }


def compute_deflated_sharpe_ratio(returns: pd.Series, *, trial_count: int) -> dict[str, float | int | None]:
    cleaned = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    sample_count = int(len(cleaned))
    if sample_count < 2:
        return {
            "dsr": None,
            "observed_sharpe": None,
            "benchmark_sharpe": None,
            "return_count": sample_count,
        }

    volatility = float(cleaned.std(ddof=0))
    if volatility <= 0:
        return {
            "dsr": None,
            "observed_sharpe": None,
            "benchmark_sharpe": None,
            "return_count": sample_count,
        }

    observed_sharpe = float((cleaned.mean() / volatility) * (252**0.5))
    sharpe_std = (1.0 / max(sample_count - 1, 1)) ** 0.5
    benchmark_sharpe = float(_expected_max_sharpe(max(1, int(trial_count)), sharpe_std))
    skew = float(cleaned.skew()) if sample_count >= 3 else 0.0
    kurtosis = float(cleaned.kurtosis()) + 3.0 if sample_count >= 4 else 3.0
    denominator = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_sharpe**2)
    if denominator <= 0:
        return {
            "dsr": None,
            "observed_sharpe": observed_sharpe,
            "benchmark_sharpe": benchmark_sharpe,
            "return_count": sample_count,
        }

    z_score = ((observed_sharpe - benchmark_sharpe) * ((sample_count - 1) ** 0.5)) / (denominator**0.5)
    return {
        "dsr": float(STANDARD_NORMAL.cdf(z_score)),
        "observed_sharpe": observed_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "return_count": sample_count,
    }


def _build_trial_fold_matrix(trial_records: list[dict[str, object]]) -> list[list[float]] | None:
    fold_scores = [
        [float(value) for value in record.get("fold_scores", [])]
        for record in trial_records
        if record.get("fold_scores")
    ]
    if not fold_scores:
        return None
    expected = len(fold_scores[0])
    if expected == 0 or any(len(scores) != expected for scores in fold_scores):
        return None
    return fold_scores


def _best_index(values: list[float], *, maximize: bool) -> int:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=maximize)
    return int(indexed[0][0])


def _rank_percentile(values: list[float], *, selected_index: int, maximize: bool) -> float:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=maximize)
    rank = next(position for position, item in enumerate(indexed, start=1) if item[0] == selected_index)
    return rank / (len(values) + 1.0)


def _logit(value: float, *, eps: float = 1e-6) -> float:
    clipped = min(1.0 - eps, max(eps, value))
    return float(log(clipped / (1.0 - clipped)))


def _expected_max_sharpe(trial_count: int, sharpe_std: float) -> float:
    if trial_count <= 1:
        return 0.0
    first_quantile = STANDARD_NORMAL.inv_cdf(1.0 - (1.0 / max(trial_count, 1)))
    second_quantile = STANDARD_NORMAL.inv_cdf(1.0 - (1.0 / (max(trial_count, 1) * e)))
    return sharpe_std * ((1.0 - EULER_GAMMA) * first_quantile + EULER_GAMMA * second_quantile)
