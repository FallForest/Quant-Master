from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.loading import load_bars_for_symbols
from data.provider_factory import build_data_provider
from ml.artifacts import save_signal_artifact
from ml.dataset import build_training_dataset, drop_rows_without_features
from ml.models import fit_model, resolve_model_params, serialize_model_params
from ml.tuning import TuningConfig, TuningResult, tune_model_params
from ml.validation import (
    WalkForwardConfig,
    ValidationSummary,
    evaluate_explicit_split,
    evaluate_holdout_validation,
    evaluate_walk_forward_validation,
)


def train_ml_signal_model(
    *,
    market: str,
    provider_name: str,
    data_root: str,
    universe_root: str,
    reference_root: str = "data/reference",
    adjust: str,
    timeframe: str,
    symbols: list[str] | None,
    universe: str | None,
    start_date: str,
    end_date: str,
    artifact_path: str,
    model_name: str = "ridge",
    model_params: dict | None = None,
    feature_columns: list[str] | None = None,
    label_horizon: int = 5,
    target_mode: str = "future_return",
    train_end_date: str | None = None,
    valid_start_date: str | None = None,
    valid_end_date: str | None = None,
    validation_mode: str = "holdout",
    walk_forward_config: WalkForwardConfig | None = None,
    tuning_config: TuningConfig | None = None,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> dict:
    selected_features = list(feature_columns or [])
    if not selected_features:
        raise ValueError("feature_columns must not be empty.")

    provider = build_data_provider(
        provider_name=provider_name,
        data_root=data_root,
        universe_root=universe_root,
        adjust=adjust,
    )
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    resolved_symbols = list(symbols or [])
    if not resolved_symbols:
        resolved_symbols = provider.load_universe(market=market, universe=universe, date=start)
    if not resolved_symbols:
        raise ValueError("No symbols were resolved for ML training.")

    data = _load_training_bars(
        provider=provider,
        market=market,
        timeframe=timeframe,
        symbols=resolved_symbols,
        start=start,
        end=end,
    )
    bundle = build_training_dataset(
        data=data,
        label_horizon=label_horizon,
        feature_columns=selected_features,
        reference_root=reference_root,
        market=market,
        target_mode=target_mode,
    )
    training_frame = drop_rows_without_features(
        frame=bundle.frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
    )
    if training_frame.empty:
        raise ValueError("Training dataset is empty after feature and label filtering.")

    resolved_model_params = resolve_model_params(model_name=model_name, model_params=model_params)
    tuning_summary = TuningResult(
        enabled=False,
        metric="",
        direction="",
        best_score=None,
        best_params=serialize_model_params(resolved_model_params),
        trials=0,
        completed_trials=0,
        trial_records=[],
    )
    selection_frame = training_frame
    selection_mode = validation_mode
    if tuning_config is not None:
        tuning_frame, selection_frame = _split_tuning_and_selection_frames(
            frame=training_frame,
            validation_mode=validation_mode,
            train_end_date=train_end_date,
            valid_start_date=valid_start_date,
            valid_end_date=valid_end_date,
            full_end_date=end_date,
        )
        tuning_summary = tune_model_params(
            frame=tuning_frame,
            feature_columns=bundle.feature_columns,
            label_column=bundle.label_column,
            model_name=model_name,
            base_model_params=model_params,
            tuning_config=tuning_config,
            validation_mode=validation_mode,
            train_end_date=train_end_date if validation_mode == "holdout" else None,
            valid_start_date=None,
            valid_end_date=None,
            walk_forward_config=walk_forward_config,
            purge_size=purge_size,
            embargo_size=embargo_size,
        )
        resolved_model_params = resolve_model_params(model_name=model_name, model_params=tuning_summary.best_params)
        validation_summary = evaluate_explicit_split(
            train_frame=tuning_frame,
            valid_frame=selection_frame,
            feature_columns=bundle.feature_columns,
            label_column=bundle.label_column,
            model_name=model_name,
            model_params=resolved_model_params,
            mode="holdout",
        )
        selection_mode = "holdout"
    else:
        validation_summary = _run_validation(
            frame=training_frame,
            feature_columns=bundle.feature_columns,
            label_column=bundle.label_column,
            model_name=model_name,
            model_params=resolved_model_params,
            validation_mode=validation_mode,
            train_end_date=train_end_date,
            valid_start_date=valid_start_date,
            valid_end_date=valid_end_date,
            walk_forward_config=walk_forward_config,
            purge_size=purge_size,
            embargo_size=embargo_size,
        )
    estimator = fit_model(
        frame=training_frame,
        feature_columns=bundle.feature_columns,
        label_column=bundle.label_column,
        model_name=model_name,
        model_params=resolved_model_params,
    )
    metadata = {
        "artifact_type": "ml_signal",
        "model_name": model_name,
        "model_params": serialize_model_params(resolved_model_params),
        "feature_columns": bundle.feature_columns,
        "label_column": bundle.label_column,
        "label_horizon": int(label_horizon),
        "target_mode": target_mode,
        "market": market,
        "provider": provider_name,
        "timeframe": timeframe,
        "adjust": adjust,
        "reference_root": reference_root,
        "symbols_count": len(resolved_symbols),
        "train_rows": int(len(training_frame)),
        "data_start": start_date,
        "data_end": end_date,
        "validation_mode": validation_mode,
        "selection_validation_mode": selection_mode,
        "validation_fold_count": validation_summary.fold_count,
        "validation_rows": validation_summary.valid_rows,
        "validation_metrics": validation_summary.metrics.as_dict(),
        "validation_folds": validation_summary.folds,
        "walk_forward_config": _serialize_walk_forward_config(walk_forward_config),
        "purge_size": int(purge_size),
        "embargo_size": int(embargo_size),
        "tuning": {
            **tuning_summary.as_dict(),
            "validation_mode": validation_mode if tuning_config is not None else None,
            "selection_valid_start": valid_start_date if tuning_config is not None else None,
            "selection_valid_end": valid_end_date if tuning_config is not None else None,
        },
    }
    save_signal_artifact(artifact_path=artifact_path, model=estimator, metadata=metadata)
    return metadata


def _run_validation(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict,
    validation_mode: str,
    train_end_date: str | None,
    valid_start_date: str | None,
    valid_end_date: str | None,
    walk_forward_config: WalkForwardConfig | None,
    purge_size: int,
    embargo_size: int,
) -> ValidationSummary:
    if validation_mode == "walk_forward":
        if walk_forward_config is None:
            raise ValueError("walk_forward_config is required when validation_mode=walk_forward.")
        return evaluate_walk_forward_validation(
            frame=frame,
            feature_columns=feature_columns,
            label_column=label_column,
            model_name=model_name,
            model_params=model_params,
            config=walk_forward_config,
        )
    return evaluate_holdout_validation(
        frame=frame,
        feature_columns=feature_columns,
        label_column=label_column,
        model_name=model_name,
        model_params=model_params,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )


def _serialize_walk_forward_config(config: WalkForwardConfig | None) -> dict[str, object] | None:
    if config is None:
        return None
    return {
        "train_size": int(config.train_size),
        "valid_size": int(config.valid_size),
        "step_size": int(config.step_size) if config.step_size is not None else None,
        "expanding": bool(config.expanding),
        "purge_size": int(config.purge_size),
        "embargo_size": int(config.embargo_size),
    }


def _split_tuning_and_selection_frames(
    *,
    frame: pd.DataFrame,
    validation_mode: str,
    train_end_date: str | None,
    valid_start_date: str | None,
    valid_end_date: str | None,
    full_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not valid_start_date or not valid_end_date:
        raise ValueError("Optuna tuning requires valid_start_date and valid_end_date as an outer selection window.")
    selection_start = pd.Timestamp(valid_start_date)
    selection_end = pd.Timestamp(valid_end_date)
    if selection_end.date().isoformat() != pd.Timestamp(full_end_date).date().isoformat():
        raise ValueError("When Optuna tuning is enabled, valid_end_date must equal end_date to form the final selection window.")
    if validation_mode == "holdout":
        if train_end_date is None:
            raise ValueError("Holdout Optuna tuning requires train_end_date for the inner tuning split.")
        if pd.Timestamp(train_end_date) >= selection_start:
            raise ValueError("train_end_date must be earlier than valid_start_date when Optuna tuning is enabled.")

    tuning_frame = frame[frame["timestamp"] < selection_start].copy().reset_index(drop=True)
    selection_frame = frame[
        (frame["timestamp"] >= selection_start) & (frame["timestamp"] <= selection_end)
    ].copy().reset_index(drop=True)
    if tuning_frame.empty:
        raise ValueError("Tuning frame is empty before the outer selection window.")
    if selection_frame.empty:
        raise ValueError("Selection frame is empty inside the requested valid_start_date/valid_end_date window.")
    return tuning_frame, selection_frame


def _load_training_bars(provider, market: str, timeframe: str, symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    return load_bars_for_symbols(
        provider=provider,
        market=market,
        timeframe=timeframe,
        symbols=symbols,
        start=start,
        end=end,
        progress_desc="Loading training bars",
        show_progress=True,
        empty_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )
