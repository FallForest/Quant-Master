from __future__ import annotations

import json
from dataclasses import dataclass
from math import log2

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SUPPORTED_MODEL_NAMES = ("ridge", "xgboost", "catboost")

RECOMMENDED_MODEL_PARAMS = {
    "ridge": {
        "alpha": 1.0,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cpu",
    },
    "catboost": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": 42,
        "thread_count": -1,
        "task_type": "CPU",
        "devices": None,
    },
}


@dataclass(slots=True)
class ValidationMetrics:
    mae: float
    r2: float
    pearson_ic: float
    spearman_ic: float
    ic_std: float
    ic_ir: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "r2": self.r2,
            "pearson_ic": self.pearson_ic,
            "spearman_ic": self.spearman_ic,
            "ic_std": self.ic_std,
            "ic_ir": self.ic_ir,
            "ndcg_at_10": self.ndcg_at_10,
            "oos_pearson_ic": self.pearson_ic,
            "oos_spearman_ic": self.spearman_ic,
            "oos_ic_std": self.ic_std,
            "oos_ic_ir": self.ic_ir,
            "oos_ndcg_at_10": self.ndcg_at_10,
        }


def recommended_model_params(model_name: str) -> dict:
    if model_name not in RECOMMENDED_MODEL_PARAMS:
        available = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
        raise ValueError(f"Unsupported ML model: {model_name}. Available models: {available}")
    return serialize_model_params(RECOMMENDED_MODEL_PARAMS[model_name])


def resolve_model_params(model_name: str, model_params: dict | None = None) -> dict:
    resolved = recommended_model_params(model_name)
    resolved.update(model_params or {})
    return resolved


def build_model(model_name: str, model_params: dict | None = None):
    params = resolve_model_params(model_name=model_name, model_params=model_params)
    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(params["alpha"]), random_state=None)),
            ]
        )
    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        reg_lambda=float(params["reg_lambda"]),
                        random_state=int(params["random_state"]),
                        n_jobs=int(params["n_jobs"]),
                        tree_method=str(params["tree_method"]),
                        device=str(params["device"]),
                        objective="reg:squarederror",
                        verbosity=0,
                    ),
                ),
            ]
        )
    if model_name == "catboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=int(params["iterations"]),
                        depth=int(params["depth"]),
                        learning_rate=float(params["learning_rate"]),
                        l2_leaf_reg=float(params["l2_leaf_reg"]),
                        loss_function=str(params["loss_function"]),
                        random_seed=int(params["random_seed"]),
                        thread_count=int(params["thread_count"]),
                        task_type=str(params["task_type"]),
                        devices=None if params.get("devices") in {None, "", "None"} else str(params["devices"]),
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    available = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
    raise ValueError(f"Unsupported ML model: {model_name}. Available models: {available}")


def fit_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict | None = None,
):
    estimator = build_model(model_name=model_name, model_params=model_params)
    estimator.fit(frame[feature_columns], frame[label_column])
    return estimator


def score_frame(estimator, frame: pd.DataFrame, feature_columns: list[str], score_column: str = "prediction") -> pd.DataFrame:
    scored = frame.copy()
    complete = scored[feature_columns].notna().all(axis=1)
    scored[score_column] = pd.NA
    if complete.any():
        predictions = estimator.predict(scored.loc[complete, feature_columns])
        scored.loc[complete, score_column] = predictions
    return scored


def evaluate_model(estimator, frame: pd.DataFrame, feature_columns: list[str], label_column: str) -> ValidationMetrics:
    if frame.empty:
        return ValidationMetrics(
            mae=0.0,
            r2=0.0,
            pearson_ic=0.0,
            spearman_ic=0.0,
            ic_std=0.0,
            ic_ir=0.0,
            ndcg_at_10=0.0,
        )
    predictions = estimator.predict(frame[feature_columns])
    actual = frame[label_column].astype(float)
    predicted = pd.Series(predictions, index=frame.index, dtype=float)
    scored = frame.assign(__prediction=predicted, __label=actual)
    pearson_series = _cross_sectional_ic_series(
        frame=scored,
        prediction_column="__prediction",
        label_column="__label",
        method="pearson",
    )
    spearman_series = _cross_sectional_ic_series(
        frame=scored,
        prediction_column="__prediction",
        label_column="__label",
        method="spearman",
    )
    pearson = float(pearson_series.mean()) if not pearson_series.empty else 0.0
    spearman = float(spearman_series.mean()) if not spearman_series.empty else 0.0
    ic_std = float(spearman_series.std(ddof=0)) if len(spearman_series) > 1 else 0.0
    ic_ir = float(spearman / ic_std) if ic_std > 0 else 0.0
    return ValidationMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        r2=float(r2_score(actual, predicted)),
        pearson_ic=0.0 if pd.isna(pearson) else float(pearson),
        spearman_ic=0.0 if pd.isna(spearman) else float(spearman),
        ic_std=0.0 if pd.isna(ic_std) else float(ic_std),
        ic_ir=0.0 if pd.isna(ic_ir) else float(ic_ir),
        ndcg_at_10=_mean_ndcg_at_k(
            frame=scored,
            prediction_column="__prediction",
            label_column="__label",
            k=10,
        ),
    )


def serialize_model_params(model_params: dict | None) -> dict:
    return json.loads(json.dumps(model_params or {}))


def aggregate_validation_metrics(metrics_list: list[ValidationMetrics]) -> ValidationMetrics:
    if not metrics_list:
        return ValidationMetrics(
            mae=0.0,
            r2=0.0,
            pearson_ic=0.0,
            spearman_ic=0.0,
            ic_std=0.0,
            ic_ir=0.0,
            ndcg_at_10=0.0,
        )
    count = float(len(metrics_list))
    return ValidationMetrics(
        mae=float(sum(item.mae for item in metrics_list) / count),
        r2=float(sum(item.r2 for item in metrics_list) / count),
        pearson_ic=float(sum(item.pearson_ic for item in metrics_list) / count),
        spearman_ic=float(sum(item.spearman_ic for item in metrics_list) / count),
        ic_std=float(sum(item.ic_std for item in metrics_list) / count),
        ic_ir=float(sum(item.ic_ir for item in metrics_list) / count),
        ndcg_at_10=float(sum(item.ndcg_at_10 for item in metrics_list) / count),
    )


def _cross_sectional_ic_series(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    label_column: str,
    method: str,
) -> pd.Series:
    rows: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("timestamp", sort=False):
        valid = group[[prediction_column, label_column]].dropna()
        if len(valid) < 2:
            continue
        corr = valid[label_column].corr(valid[prediction_column], method=method)
        if pd.isna(corr):
            continue
        rows.append({"timestamp": timestamp, "value": float(corr)})
    if not rows:
        return pd.Series(dtype=float)
    series = pd.DataFrame(rows).set_index("timestamp")["value"].astype(float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _mean_ndcg_at_k(frame: pd.DataFrame, *, prediction_column: str, label_column: str, k: int) -> float:
    scores: list[float] = []
    for _, group in frame.groupby("timestamp", sort=False):
        if group.empty:
            continue
        predicted = group.sort_values(prediction_column, ascending=False).head(k)
        ideal = group.sort_values(label_column, ascending=False).head(k)
        dcg = _dcg(predicted[label_column].tolist())
        idcg = _dcg(ideal[label_column].tolist())
        if idcg > 0:
            scores.append(float(dcg / idcg))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _dcg(values: list[float]) -> float:
    total = 0.0
    for index, value in enumerate(values, start=1):
        gain = (2.0 ** float(value)) - 1.0
        total += gain / log2(index + 1)
    return float(total)
