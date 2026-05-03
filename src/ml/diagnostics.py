from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, log
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from ml.features import build_technical_features
from ml.labels import (
    add_cross_sectional_rank_label,
    add_future_return_label,
    future_rank_label_name,
    future_return_label_name,
)
from ml.models import evaluate_model, evaluate_scored_frame, score_frame


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


def build_ic_decay_profile(
    *,
    estimator,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_mode: str,
    horizons: list[int],
) -> list[dict[str, object]]:
    normalized_horizons = []
    for horizon in horizons:
        value = int(horizon)
        if value < 1 or value in normalized_horizons:
            continue
        normalized_horizons.append(value)

    rows: list[dict[str, object]] = []
    for horizon in normalized_horizons:
        labeled, label_column = _label_decay_frame(
            frame=frame,
            target_mode=target_mode,
            horizon=horizon,
        )
        labeled = labeled.dropna(subset=[*feature_columns, label_column]).reset_index(drop=True)
        if labeled.empty:
            rows.append(
                {
                    "horizon": horizon,
                    "rows": 0,
                    "ic": 0.0,
                    "rank_ic": 0.0,
                    "ic_std": 0.0,
                    "rank_ic_std": 0.0,
                    "icir": 0.0,
                    "rank_icir": 0.0,
                    "ndcg_at_10": 0.0,
                }
            )
            continue
        metrics = evaluate_model(
            estimator=estimator,
            frame=labeled,
            feature_columns=feature_columns,
            label_column=label_column,
        )
        rows.append(
            {
                "horizon": horizon,
                "rows": int(len(labeled)),
                **metrics.as_dict(),
            }
        )
    return rows


def build_signal_slice_diagnostics(
    *,
    estimator,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    reference_root: str,
    market: str,
    benchmark_symbol: str = "sh000300",
    industry_standard: str = "申银万国行业分类标准",
    market_cap_bucket_count: int = 5,
) -> dict[str, object]:
    if frame.empty:
        return {
            "year_windows": [],
            "market_style_regimes": [],
            "industry_buckets": [],
            "market_cap_buckets": [],
            "meta": {
                "benchmark_symbol": benchmark_symbol,
                "industry_standard": industry_standard,
                "market_cap_bucket_count": int(market_cap_bucket_count),
                "industry_coverage_ratio": 0.0,
            },
        }

    scored = score_frame(estimator=estimator, frame=frame, feature_columns=feature_columns, score_column="prediction")
    scored["label"] = frame[label_column].astype(float)
    scored = scored.dropna(subset=["prediction", "label"]).reset_index(drop=True)
    if scored.empty:
        return {
            "year_windows": [],
            "market_style_regimes": [],
            "industry_buckets": [],
            "market_cap_buckets": [],
            "meta": {
                "benchmark_symbol": benchmark_symbol,
                "industry_standard": industry_standard,
                "market_cap_bucket_count": int(market_cap_bucket_count),
                "industry_coverage_ratio": 0.0,
            },
        }

    year_windows = _build_year_window_diagnostics(scored)
    market_style_regimes = _build_market_style_diagnostics(
        scored=scored,
        reference_root=reference_root,
        market=market,
        benchmark_symbol=benchmark_symbol,
    )
    industry_frame = _attach_industry_labels(
        frame=scored,
        reference_root=reference_root,
        market=market,
        preferred_standard=industry_standard,
    )
    industry_buckets = _build_bucket_diagnostics(frame=industry_frame, group_column="industry_level_1")
    market_cap_frame = _attach_market_cap_buckets(
        frame=scored,
        reference_root=reference_root,
        market=market,
        bucket_count=market_cap_bucket_count,
    )
    market_cap_buckets = _build_bucket_diagnostics(frame=market_cap_frame, group_column="market_cap_bucket")
    industry_coverage_ratio = float(industry_frame["industry_level_1"].notna().mean()) if not industry_frame.empty else 0.0
    return {
        "year_windows": year_windows,
        "market_style_regimes": market_style_regimes,
        "industry_buckets": industry_buckets,
        "market_cap_buckets": market_cap_buckets,
        "meta": {
            "benchmark_symbol": benchmark_symbol,
            "industry_standard": industry_standard,
            "market_cap_bucket_count": int(market_cap_bucket_count),
            "industry_coverage_ratio": industry_coverage_ratio,
        },
    }


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


def _label_decay_frame(*, frame: pd.DataFrame, target_mode: str, horizon: int) -> tuple[pd.DataFrame, str]:
    if target_mode == "cross_sectional_rank":
        labeled = add_cross_sectional_rank_label(frame, horizon=horizon)
        return labeled, future_rank_label_name(horizon)
    labeled = add_future_return_label(frame, horizon=horizon)
    return labeled, future_return_label_name(horizon)


def _build_year_window_diagnostics(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(frame["timestamp"].dt.year, sort=True):
        metrics = evaluate_scored_frame(frame=group, prediction_column="prediction", label_column="label")
        start = pd.to_datetime(group["timestamp"]).min().date().isoformat()
        end = pd.to_datetime(group["timestamp"]).max().date().isoformat()
        suffix = "" if end.endswith("-12-31") else "_ytd"
        rows.append(
            {
                "name": f"oos_{year}{suffix}",
                "start_date": start,
                "end_date": end,
                "rows": int(len(group)),
                "metrics": metrics.as_dict(),
            }
        )
    return rows


def _build_market_style_diagnostics(
    *,
    scored: pd.DataFrame,
    reference_root: str,
    market: str,
    benchmark_symbol: str,
) -> list[dict[str, object]]:
    benchmark_path = Path(reference_root) / market / "index" / f"{benchmark_symbol}.csv"
    if not benchmark_path.exists():
        return []
    benchmark = pd.read_csv(benchmark_path)
    timestamp_column = "timestamp" if "timestamp" in benchmark.columns else "date"
    if timestamp_column not in benchmark.columns or "close" not in benchmark.columns:
        return []
    regime = benchmark.rename(columns={timestamp_column: "timestamp"}).copy()
    regime["timestamp"] = pd.to_datetime(regime["timestamp"])
    regime["benchmark_return_20"] = pd.to_numeric(regime["close"], errors="coerce").pct_change(20)
    regime["benchmark_vol_20"] = pd.to_numeric(regime["close"], errors="coerce").pct_change().rolling(20, min_periods=20).std(ddof=0)
    merged = scored.merge(regime[["timestamp", "benchmark_return_20", "benchmark_vol_20"]], on="timestamp", how="left")
    valid = merged.dropna(subset=["benchmark_return_20", "benchmark_vol_20"]).copy()
    if valid.empty:
        return []
    return_cutoff = float(valid["benchmark_return_20"].median())
    vol_cutoff = float(valid["benchmark_vol_20"].median())
    valid["market_style_regime"] = valid.apply(
        lambda row: _market_style_label(
            return_20=float(row["benchmark_return_20"]),
            volatility_20=float(row["benchmark_vol_20"]),
            return_cutoff=return_cutoff,
            vol_cutoff=vol_cutoff,
        ),
        axis=1,
    )
    return _build_bucket_diagnostics(frame=valid, group_column="market_style_regime")


def _market_style_label(*, return_20: float, volatility_20: float, return_cutoff: float, vol_cutoff: float) -> str:
    trend = "trend_up" if return_20 >= return_cutoff else "trend_down"
    volatility = "high_vol" if volatility_20 >= vol_cutoff else "low_vol"
    return f"{trend}_{volatility}"


def _attach_market_cap_buckets(
    *,
    frame: pd.DataFrame,
    reference_root: str,
    market: str,
    bucket_count: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(market_cap_bucket=pd.Series(dtype="object"))
    if "log_total_mkt_cap" in frame.columns:
        enriched = frame.copy()
    else:
        base = frame[[column for column in ["timestamp", "symbol", "open", "high", "low", "close", "volume"] if column in frame.columns]].copy()
        base = build_technical_features(
            data=base,
            factor_names=["log_total_mkt_cap"],
            reference_root=reference_root,
            market=market,
        )
        enriched = frame.merge(base[["timestamp", "symbol", "log_total_mkt_cap"]], on=["timestamp", "symbol"], how="left")

    def _bucket_for_group(group: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(group["log_total_mkt_cap"], errors="coerce")
        valid = values.dropna()
        result = pd.Series(pd.NA, index=group.index, dtype="object")
        if valid.empty:
            return result
        rank = valid.rank(method="first", pct=True)
        bucket_ids = (rank * bucket_count).apply(lambda value: min(bucket_count, max(1, int(value if value == int(value) else int(value) + 1))))
        labels = bucket_ids.map(
            lambda bucket: (
                f"cap_q{bucket}_smallest"
                if bucket == 1
                else f"cap_q{bucket}_largest"
                if bucket == bucket_count
                else f"cap_q{bucket}"
            )
        )
        result.loc[valid.index] = labels
        return result

    enriched["market_cap_bucket"] = enriched.groupby("timestamp", sort=False, group_keys=False).apply(_bucket_for_group)
    return enriched


def _attach_industry_labels(
    *,
    frame: pd.DataFrame,
    reference_root: str,
    market: str,
    preferred_standard: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(industry_level_1=pd.Series(dtype="object"))
    if "industry_level_1" in frame.columns:
        enriched = frame.copy()
        if "sector" in enriched.columns:
            enriched["industry_level_1"] = enriched["industry_level_1"].fillna(enriched["sector"])
        return enriched
    frames: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        industry_path = Path(reference_root) / market / "industry" / f"{symbol}.csv"
        if not industry_path.exists():
            item = group.copy()
            item["industry_level_1"] = pd.NA
            frames.append(item)
            continue
        reference = pd.read_csv(industry_path)
        if reference.empty or "change_date" not in reference.columns:
            item = group.copy()
            item["industry_level_1"] = pd.NA
            frames.append(item)
            continue
        reference["change_date"] = pd.to_datetime(reference["change_date"])
        preferred = reference[reference["standard"] == preferred_standard].copy()
        active_reference = preferred if not preferred.empty else reference.copy()
        if "symbol" in active_reference.columns:
            active_reference = active_reference.drop(columns=["symbol"])
        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            active_reference.sort_values("change_date"),
            left_on="timestamp",
            right_on="change_date",
            direction="backward",
        )
        merged["industry_level_1"] = merged["industry_level_1"].fillna(merged.get("sector"))
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _build_bucket_diagnostics(*, frame: pd.DataFrame, group_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = frame.dropna(subset=[group_column]).copy()
    if valid.empty:
        return rows
    for group_name, item in valid.groupby(group_column, sort=True):
        metrics = evaluate_scored_frame(frame=item, prediction_column="prediction", label_column="label")
        rows.append(
            {
                "name": str(group_name),
                "rows": int(len(item)),
                "metrics": metrics.as_dict(),
            }
        )
    rows.sort(key=lambda item: (-int(item["rows"]), str(item["name"])))
    return rows
