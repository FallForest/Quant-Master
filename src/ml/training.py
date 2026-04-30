from __future__ import annotations

import pandas as pd

from ml.artifacts import save_signal_artifact
from ml.models import fit_model, resolve_model_params, serialize_model_params
from ml.prepared_data import PreparedSignalDataset, SignalDatasetCache, prepare_signal_dataset
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
    feature_normalization: str = "none",
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
    dataset_cache: SignalDatasetCache | None = None,
    prepared_dataset: PreparedSignalDataset | None = None,
) -> dict:
    """训练一个 ML 信号模型，并把模型与元数据保存为 artifact。

    这是训练链路的核心函数，负责：
    1. 准备数据集
    2. 运行验证 / 调参
    3. 在完整训练集上拟合最终模型
    4. 产出 artifact metadata，供后续 signal_test / backfill / compare 复用
    """

    # prepared_dataset 允许上游把已经构造好的数据直接传进来，
    # 避免候选筛选后重复加载和重复做特征工程。
    prepared = prepared_dataset or prepare_signal_dataset(
        market=market,
        provider_name=provider_name,
        data_root=data_root,
        universe_root=universe_root,
        reference_root=reference_root,
        adjust=adjust,
        timeframe=timeframe,
        symbols=symbols,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns or []),
        feature_normalization=feature_normalization,
        label_horizon=label_horizon,
        target_mode=target_mode,
        progress_desc="Loading training bars",
        dataset_cache=dataset_cache,
    )
    training_frame = prepared.frame
    if training_frame.empty:
        raise ValueError("Training dataset is empty after feature and label filtering.")

    resolved_model_params = resolve_model_params(model_name=model_name, model_params=model_params)
    # 默认先构造一个“未启用 tuning”的占位 summary；
    # 如果后面真的启用了 tuning，会被真实结果覆盖。
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
        # tuning 和最终 selection 是两层不同目的的切分：
        # - tuning_frame: 用来搜索参数
        # - selection_frame: 用来从候选参数中挑最终方案
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
            feature_columns=prepared.feature_columns,
            label_column=prepared.label_column,
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
            feature_columns=prepared.feature_columns,
            label_column=prepared.label_column,
            model_name=model_name,
            model_params=resolved_model_params,
            mode="holdout",
        )
        selection_mode = "holdout"
    else:
        validation_summary = _run_validation(
            frame=training_frame,
            feature_columns=prepared.feature_columns,
            label_column=prepared.label_column,
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
    # 完成验证/调参后，最终模型始终在完整 training_frame 上重新拟合一次。
    estimator = fit_model(
        frame=training_frame,
        feature_columns=prepared.feature_columns,
        label_column=prepared.label_column,
        model_name=model_name,
        model_params=resolved_model_params,
    )
    # metadata 是后续所有流程的桥梁：
    # signal_test / backfill / compare 都依赖它复原训练口径。
    metadata = {
        "artifact_type": "ml_signal",
        "model_name": model_name,
        "model_params": serialize_model_params(resolved_model_params),
        "feature_columns": prepared.feature_columns,
        "feature_normalization": feature_normalization,
        "label_column": prepared.label_column,
        "label_horizon": int(label_horizon),
        "target_mode": target_mode,
        "market": market,
        "provider": provider_name,
        "timeframe": timeframe,
        "adjust": adjust,
        "reference_root": reference_root,
        "symbols_count": len(prepared.symbols),
        "train_rows": int(len(training_frame)),
        "data_preparation": dict(prepared.diagnostics),
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
    """根据配置选择 holdout 或 walk-forward 验证。"""

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
    """把 walk-forward 配置转成可写入 artifact 的 JSON 结构。"""

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
    """把完整训练窗拆成“调参内层样本”和“最终选择外层样本”。

    这样做的目的是避免直接拿 tuning 过程中见过的数据决定最终候选，
    减少参数选择阶段的信息泄漏。
    """

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
