from __future__ import annotations

import json
from dataclasses import dataclass
from math import log2

import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    SimpleImputer = None
    Ridge = None
    mean_absolute_error = None
    r2_score = None
    Pipeline = None
    StandardScaler = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CatBoostRegressor = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None

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
    ic: float
    rank_ic: float
    ic_std: float
    rank_ic_std: float
    icir: float
    rank_icir: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "r2": self.r2,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "ic_std": self.ic_std,
            "rank_ic_std": self.rank_ic_std,
            "icir": self.icir,
            "rank_icir": self.rank_icir,
            "ndcg_at_10": self.ndcg_at_10,
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
    _require_base_ml_dependencies()
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
        _require_xgboost()
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
        _require_catboost()
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
    _require_base_ml_dependencies()
    if frame.empty:
        return _empty_validation_metrics()
    scored = score_frame(estimator=estimator, frame=frame, feature_columns=feature_columns, score_column="__prediction")
    scored["__label"] = frame[label_column].astype(float)
    scored = scored.dropna(subset=["__prediction", "__label"]).reset_index(drop=True)
    return evaluate_scored_frame(
        frame=scored,
        prediction_column="__prediction",
        label_column="__label",
    )


def evaluate_scored_frame(
    *,
    frame: pd.DataFrame,
    prediction_column: str,
    label_column: str,
) -> ValidationMetrics:
    _require_base_ml_dependencies()
    if frame.empty:
        return _empty_validation_metrics()
    actual = frame[label_column].astype(float)
    predicted = frame[prediction_column].astype(float)
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
    pearson_std = float(pearson_series.std(ddof=0)) if len(pearson_series) > 1 else 0.0
    spearman_std = float(spearman_series.std(ddof=0)) if len(spearman_series) > 1 else 0.0
    pearson_ir = float(pearson / pearson_std) if pearson_std > 0 else 0.0
    spearman_ir = float(spearman / spearman_std) if spearman_std > 0 else 0.0
    return ValidationMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        r2=float(r2_score(actual, predicted)),
        ic=0.0 if pd.isna(pearson) else float(pearson),
        rank_ic=0.0 if pd.isna(spearman) else float(spearman),
        ic_std=0.0 if pd.isna(pearson_std) else float(pearson_std),
        rank_ic_std=0.0 if pd.isna(spearman_std) else float(spearman_std),
        icir=0.0 if pd.isna(pearson_ir) else float(pearson_ir),
        rank_icir=0.0 if pd.isna(spearman_ir) else float(spearman_ir),
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
        return _empty_validation_metrics()
    count = float(len(metrics_list))
    return ValidationMetrics(
        mae=float(sum(item.mae for item in metrics_list) / count),
        r2=float(sum(item.r2 for item in metrics_list) / count),
        ic=float(sum(item.ic for item in metrics_list) / count),
        rank_ic=float(sum(item.rank_ic for item in metrics_list) / count),
        ic_std=float(sum(item.ic_std for item in metrics_list) / count),
        rank_ic_std=float(sum(item.rank_ic_std for item in metrics_list) / count),
        icir=float(sum(item.icir for item in metrics_list) / count),
        rank_icir=float(sum(item.rank_icir for item in metrics_list) / count),
        ndcg_at_10=float(sum(item.ndcg_at_10 for item in metrics_list) / count),
    )


def _require_base_ml_dependencies() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is not installed. Run `.venv\\Scripts\\python -m pip install -r requirements.txt` first."
        ) from _SKLEARN_IMPORT_ERROR


def _require_xgboost() -> None:
    if _XGBOOST_IMPORT_ERROR is not None:
        raise RuntimeError(
            "xgboost is not installed. Run `.venv\\Scripts\\python -m pip install -r requirements.txt` first."
        ) from _XGBOOST_IMPORT_ERROR


def _require_catboost() -> None:
    if _CATBOOST_IMPORT_ERROR is not None:
        raise RuntimeError(
            "catboost is not installed. Run `.venv\\Scripts\\python -m pip install -r requirements.txt` first."
        ) from _CATBOOST_IMPORT_ERROR


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


def _empty_validation_metrics() -> ValidationMetrics:
    return ValidationMetrics(
        mae=0.0,
        r2=0.0,
        ic=0.0,
        rank_ic=0.0,
        ic_std=0.0,
        rank_ic_std=0.0,
        icir=0.0,
        rank_icir=0.0,
        ndcg_at_10=0.0,
    )
