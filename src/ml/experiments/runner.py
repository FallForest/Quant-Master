from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data.thread_parallel import run_bounded_thread_pool
from ml.artifacts import load_signal_artifact, save_signal_artifact
from ml.diagnostics import build_ic_decay_profile, build_overfit_diagnostics, build_signal_slice_diagnostics
from ml.experiments.group_tracking import GroupRunTracker
from ml.experiments.loader import load_experiment_group_spec, load_experiment_spec
from ml.experiments.scheduler import GroupExecutionOptions, execute_group
from ml.experiments.specs import ExperimentGroupSpec, ExperimentSpec
from ml.factors import estimate_factor_history_lookback
from ml.models import evaluate_model, fit_model
from ml.prepared_data import PreparedSignalDataset, SignalDatasetCache, prepare_signal_dataset
from ml.runtime import get_runtime_inventory, model_uses_gpu
from ml.selection import score_candidate_metrics
from ml.tuning import TuningConfig
from ml.training import train_ml_signal_model
from ml.validation import WalkForwardConfig


def run_experiment_from_path(path: str | Path) -> dict[str, object]:
    """从 YAML 路径加载并运行单个实验。"""

    spec = load_experiment_spec(path)
    return run_experiment(spec=spec, experiment_path=Path(path))


def run_experiment(spec: ExperimentSpec, *, experiment_path: Path | None = None) -> dict[str, object]:
    """运行单个实验的完整链路。"""

    _validate_experiment_windows(spec)
    artifact_path = Path(spec.report.artifact_path or Path("artifacts") / "experiments" / spec.name / "artifact")
    report_dir = Path(spec.report.output_dir or Path("reports") / "experiments" / spec.name)
    dataset_cache = SignalDatasetCache()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 第一次训练先得到基础 artifact 和 validation metadata。
    initial_metadata = train_ml_signal_model(
        market=spec.market,
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        reference_root=spec.reference_root,
        adjust=spec.adjust,
        timeframe=spec.timeframe,
        symbols=spec.symbols or None,
        universe=spec.universe,
        start_date=spec.train.start_date,
        end_date=spec.train.end_date,
        artifact_path=str(artifact_path),
        model_name=spec.model,
        model_params=spec.model_params,
        feature_columns=spec.features,
        feature_normalization=spec.feature_normalization,
        label_horizon=spec.train.label_horizon,
        target_mode=spec.train.target_mode,
        train_end_date=spec.train.train_end_date,
        valid_start_date=spec.train.valid_start_date,
        valid_end_date=spec.train.valid_end_date,
        validation_mode=spec.train.validation_mode,
        walk_forward_config=_build_walk_forward_config(spec),
        tuning_config=_build_tuning_config(spec),
        purge_size=spec.train.purge_size,
        embargo_size=spec.train.embargo_size,
        dataset_cache=dataset_cache,
    )

    training_metadata = initial_metadata
    if spec.train.tuning:
        # tuning 只负责给出一批候选参数；真正进入最终实验 summary 的，
        # 是 candidate selection 选中的那一组参数。
        candidate_selection = _run_candidate_selection(
            spec=spec,
            training_metadata=initial_metadata,
            dataset_cache=dataset_cache,
        )
        selected_params = dict(candidate_selection["selected_model_params"])
        selection_start = str(spec.train.valid_start_date)
        pre_selection_end = (pd.Timestamp(selection_start) - pd.Timedelta(days=1)).date().isoformat()
        # 这里直接复用完整训练集缓存，避免选完候选后再重复做一遍数据准备。
        prepared_training_dataset = prepare_signal_dataset(
            market=spec.market,
            provider_name=spec.provider,
            data_root=spec.data_root,
            universe_root=spec.universe_root,
            reference_root=spec.reference_root,
            adjust=spec.adjust,
            timeframe=spec.timeframe,
            symbols=spec.symbols or None,
            universe=spec.universe,
            start_date=spec.train.start_date,
            end_date=spec.train.end_date,
            feature_columns=spec.features,
            feature_normalization=spec.feature_normalization,
            label_horizon=spec.train.label_horizon,
            target_mode=spec.train.target_mode,
            progress_desc="Loading training bars",
            dataset_cache=dataset_cache,
        )
        # 用筛出来的参数在完整训练窗上重训，产出最终 artifact。
        training_metadata = train_ml_signal_model(
            market=spec.market,
            provider_name=spec.provider,
            data_root=spec.data_root,
            universe_root=spec.universe_root,
            reference_root=spec.reference_root,
            adjust=spec.adjust,
            timeframe=spec.timeframe,
            symbols=spec.symbols or None,
            universe=spec.universe,
            start_date=spec.train.start_date,
            end_date=spec.train.end_date,
            artifact_path=str(artifact_path),
            model_name=spec.model,
            model_params=selected_params,
            feature_columns=spec.features,
            feature_normalization=spec.feature_normalization,
            label_horizon=spec.train.label_horizon,
            target_mode=spec.train.target_mode,
            train_end_date=pre_selection_end,
            valid_start_date=spec.train.valid_start_date,
            valid_end_date=spec.train.valid_end_date,
            validation_mode="holdout",
            walk_forward_config=None,
            tuning_config=None,
            purge_size=spec.train.purge_size,
            embargo_size=spec.train.embargo_size,
            dataset_cache=dataset_cache,
            prepared_dataset=prepared_training_dataset,
        )
        training_metadata["tuning"] = dict(candidate_selection["tuning"])
        training_metadata["candidate_selection"] = dict(candidate_selection["candidate_selection"])
        _rewrite_artifact_metadata(artifact_path=artifact_path, metadata=training_metadata)

    training_metadata = _enrich_training_metadata(spec=spec, metadata=training_metadata)
    _rewrite_artifact_metadata(artifact_path=artifact_path, metadata=training_metadata)

    # 完成训练后，统一从 artifact 反向加载模型来做 OOS 评估，
    # 保证评估口径和真正落盘的模型完全一致。
    signal_test = _run_signal_test(spec=spec, artifact_path=artifact_path, dataset_cache=dataset_cache)
    signal_window_metrics = _run_signal_windows(spec=spec, artifact_path=artifact_path, dataset_cache=dataset_cache)
    signal_test_dataset = _build_signal_frame(
        spec=spec,
        metadata=training_metadata,
        start_date=spec.signal_test.start_date,
        end_date=spec.signal_test.end_date,
        dataset_cache=dataset_cache,
    )
    estimator, _ = load_signal_artifact(artifact_path)
    signal_test_ic_decay = build_ic_decay_profile(
        estimator=estimator,
        frame=signal_test_dataset.frame,
        feature_columns=signal_test_dataset.feature_columns,
        target_mode=str(training_metadata.get("target_mode", spec.train.target_mode)),
        horizons=list(spec.ic_decay_horizons),
    )
    signal_test_slice_diagnostics = build_signal_slice_diagnostics(
        estimator=estimator,
        frame=signal_test_dataset.frame,
        feature_columns=signal_test_dataset.feature_columns,
        label_column=signal_test_dataset.label_column,
        reference_root=str(training_metadata.get("reference_root", spec.reference_root)),
        market=str(training_metadata.get("market", spec.market)),
        benchmark_symbol=str(training_metadata.get("benchmark_symbol", spec.benchmark_symbol)),
        industry_standard=str(training_metadata.get("industry_standard", spec.industry_standard)),
        market_cap_bucket_count=int(training_metadata.get("market_cap_bucket_count", spec.market_cap_bucket_count)),
    )
    final_diagnostics = build_overfit_diagnostics(
        trial_records=list(dict(training_metadata.get("tuning", {})).get("trial_records", [])),
        direction=str(dict(training_metadata.get("tuning", {})).get("direction", "maximize")),
        returns=pd.Series(dtype=float),
    )

    # summary 是实验层面的总出口，也是 group 汇总、对比分析、后续复盘的基础输入。
    summary = {
        "name": spec.name,
        "group": spec.group,
        "experiment_path": str(experiment_path) if experiment_path else None,
        "research_profile": spec.research_profile,
        "artifact_path": str(artifact_path),
        "report_dir": str(report_dir),
        "market": spec.market,
        "provider": spec.provider,
        "timeframe": spec.timeframe,
        "adjust": spec.adjust,
        "symbols": list(spec.symbols),
        "universe": spec.universe,
        "benchmark_symbol": spec.benchmark_symbol,
        "industry_standard": spec.industry_standard,
        "market_cap_bucket_count": int(spec.market_cap_bucket_count),
        "baseline_manifest_path": spec.baseline_manifest_path,
        "reference_root": spec.reference_root,
        "features": list(spec.features),
        "feature_normalization": spec.feature_normalization,
        "ic_decay_horizons": list(spec.ic_decay_horizons),
        "model": spec.model,
        "model_params": dict(training_metadata.get("model_params", spec.model_params)),
        "train": asdict(spec.train),
        "signal_test": asdict(spec.signal_test),
        "signal_windows": [asdict(item) for item in spec.signal_windows],
        "training_metadata": training_metadata,
        "validation_metrics": dict(training_metadata.get("validation_metrics", {})),
        "candidate_selection": dict(training_metadata.get("candidate_selection", {})),
        "signal_test_metrics": dict(signal_test["metrics"]),
        "signal_test_rows": int(signal_test["rows"]),
        "signal_windows_metrics": signal_window_metrics,
        "signal_test_ic_decay": signal_test_ic_decay,
        "signal_test_slice_diagnostics": signal_test_slice_diagnostics,
        "research_diagnostics": final_diagnostics.as_dict(),
    }
    (report_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_experiment_group_from_path(
    path: str | Path,
    *,
    continue_on_error: bool = False,
    execution_options: GroupExecutionOptions | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """从 group YAML 路径加载并执行实验组。"""

    group_path = Path(path)
    group_spec = load_experiment_group_spec(group_path)
    base_dir = group_path.parent
    experiment_paths = [str((base_dir / item).resolve()) if not Path(item).is_absolute() else item for item in group_spec.experiments]
    return run_experiment_group(
        group_spec=group_spec,
        experiment_paths=experiment_paths,
        continue_on_error=continue_on_error,
        execution_options=execution_options,
        resume=resume,
    )


def run_experiment_group(
    *,
    group_spec: ExperimentGroupSpec,
    experiment_paths: list[str],
    continue_on_error: bool = False,
    execution_options: GroupExecutionOptions | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """运行实验组，并持续把进度写入 group tracker。"""

    group_output_dir = Path(group_spec.output_dir or Path("reports") / "experiments" / "groups" / group_spec.name)
    tracker = GroupRunTracker(
        group_name=group_spec.name,
        experiment_paths=[str(path) for path in experiment_paths],
        output_dir=group_output_dir,
        mode="run",
        continue_on_error=continue_on_error,
        resume=resume,
        execution_options=_serialize_execution_options(execution_options or GroupExecutionOptions()),
    )
    # tracker 会根据已有 group_summary / manifest 判断哪些任务还没完成。
    pending_paths = tracker.pending_paths()
    if not pending_paths:
        return tracker.finalize(status="completed")

    try:
        execute_group(
            experiment_paths=pending_paths,
            continue_on_error=continue_on_error,
            options=execution_options or GroupExecutionOptions(),
            on_task_complete=lambda experiment_path, summary, error: _track_group_task(
                tracker=tracker,
                experiment_path=experiment_path,
                summary=summary,
                error=error,
            ),
        )
    except Exception:
        tracker.finalize(status="failed")
        raise

    final_status = "completed" if tracker.failed_count == 0 else "partial"
    return tracker.finalize(status=final_status)


def _build_walk_forward_config(spec: ExperimentSpec) -> WalkForwardConfig | None:
    """把实验配置里的 walk_forward 字段转换成运行时对象。"""

    if spec.train.walk_forward is None:
        return None
    return WalkForwardConfig(
        train_size=spec.train.walk_forward.train_size,
        valid_size=spec.train.walk_forward.valid_size,
        step_size=spec.train.walk_forward.step_size,
        expanding=spec.train.walk_forward.expanding,
        purge_size=spec.train.walk_forward.purge_size,
        embargo_size=spec.train.walk_forward.embargo_size,
    )


def _build_tuning_config(spec: ExperimentSpec) -> TuningConfig | None:
    """把实验配置里的 tuning 字段转换成运行时对象。"""

    if spec.train.tuning is None:
        return None
    return TuningConfig(
        trials=spec.train.tuning.trials,
        metric=spec.train.tuning.metric,
        direction=spec.train.tuning.direction,
        timeout_seconds=spec.train.tuning.timeout_seconds,
        seed=spec.train.tuning.seed,
        keep_top_trials=spec.train.tuning.keep_top_trials,
        parallel_jobs=spec.train.tuning.parallel_jobs,
        gpu_devices=list(spec.train.tuning.gpu_devices),
        cpu_threads_per_trial=spec.train.tuning.cpu_threads_per_trial,
    )


def _run_candidate_selection(
    *,
    spec: ExperimentSpec,
    training_metadata: dict[str, object],
    dataset_cache: SignalDatasetCache | None = None,
) -> dict[str, object]:
    """在 tuning 产出的候选参数中选择最终模型参数。"""

    tuning_summary = dict(training_metadata.get("tuning", {}))
    trial_records = list(tuning_summary.get("trial_records", []))
    if not trial_records:
        return {
            "selected_model_params": dict(training_metadata.get("model_params", {})),
            "candidate_selection": {"enabled": False, "candidates": [], "selected_metric": spec.train.candidate_selection.metric},
            "tuning": tuning_summary,
        }

    selection_start = str(spec.train.valid_start_date)
    selection_end = str(spec.train.valid_end_date)
    pre_selection_end = (pd.Timestamp(selection_start) - pd.Timedelta(days=1)).date().isoformat()
    # train_dataset 用于拟合候选模型；
    # selection_dataset 用于比较这些候选模型在真正 OOS 风格窗口上的表现。
    train_dataset = prepare_signal_dataset(
        market=spec.market,
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        reference_root=spec.reference_root,
        adjust=spec.adjust,
        timeframe=spec.timeframe,
        symbols=spec.symbols or None,
        universe=spec.universe,
        start_date=spec.train.start_date,
        end_date=pre_selection_end,
        feature_columns=spec.features,
        feature_normalization=spec.feature_normalization,
        label_horizon=spec.train.label_horizon,
        target_mode=spec.train.target_mode,
        progress_desc="Loading training bars",
        dataset_cache=dataset_cache,
    )
    selection_dataset = prepare_signal_dataset(
        market=spec.market,
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        reference_root=spec.reference_root,
        adjust=spec.adjust,
        timeframe=spec.timeframe,
        symbols=spec.symbols or None,
        universe=spec.universe,
        start_date=selection_start,
        end_date=selection_end,
        feature_columns=spec.features,
        feature_normalization=spec.feature_normalization,
        label_horizon=spec.train.label_horizon,
        target_mode=spec.train.target_mode,
        progress_desc="Evaluating signal window",
        dataset_cache=dataset_cache,
        # selection 窗口如果包含长 lookback 因子，需要额外预热历史样本。
        history_padding_days=estimate_factor_history_lookback(spec.features),
        trim_to_requested_range=True,
    )
    if train_dataset.label_column != selection_dataset.label_column:
        raise ValueError("Candidate selection datasets must share the same label column.")

    candidate_rows: list[dict[str, object]] = []
    max_workers = _resolve_candidate_selection_workers(
        spec=spec,
        tuning_summary=tuning_summary,
        candidate_count=len(trial_records),
    )
    if max_workers <= 1 or len(trial_records) <= 1:
        candidate_rows = [
            _evaluate_candidate_record(
                candidate_index=index,
                trial_record=record,
                spec=spec,
                train_dataset=train_dataset,
                selection_dataset=selection_dataset,
            )
            for index, record in enumerate(trial_records, start=1)
        ]
    else:
        candidate_items = list(enumerate(trial_records, start=1))
        report = run_bounded_thread_pool(
            items=candidate_items,
            max_workers=max_workers,
            submitter=lambda item: (
                lambda: _evaluate_candidate_record(
                    candidate_index=item[0],
                    trial_record=item[1],
                    spec=spec,
                    train_dataset=train_dataset,
                    selection_dataset=selection_dataset,
                ),
                None,
            ),
        )
        candidate_rows = [outcome.value for outcome in report.outcomes if outcome.value is not None]

    selection_config = spec.train.candidate_selection
    ranked_rows = sorted(
        candidate_rows,
        key=lambda item: score_candidate_metrics(
            item["selection_signal_metrics"],
            metric=selection_config.metric,
            direction=selection_config.direction,
        ),
        reverse=True,
    )[: max(1, selection_config.top_k)]
    selected = ranked_rows[0]
    return {
        "selected_model_params": dict(selected["model_params"]),
        "candidate_selection": {
            "enabled": True,
            "config": selection_config.as_dict(),
            "candidates": ranked_rows,
            "selected_candidate_index": selected["candidate_index"],
            "selected_metric": selection_config.metric,
        },
        "tuning": tuning_summary,
    }


def _rewrite_artifact_metadata(*, artifact_path: Path, metadata: dict[str, object]) -> None:
    """只更新 artifact metadata，不改动已训练好的模型对象。"""

    model, _ = load_signal_artifact(artifact_path)
    save_signal_artifact(artifact_path=artifact_path, model=model, metadata=metadata)


def _enrich_training_metadata(*, spec: ExperimentSpec, metadata: dict[str, object]) -> dict[str, object]:
    enriched = dict(metadata)
    enriched["research_profile"] = spec.research_profile
    enriched["benchmark_symbol"] = spec.benchmark_symbol
    enriched["industry_standard"] = spec.industry_standard
    enriched["market_cap_bucket_count"] = int(spec.market_cap_bucket_count)
    enriched["baseline_manifest_path"] = spec.baseline_manifest_path
    return enriched


def _run_signal_test(
    *,
    spec: ExperimentSpec,
    artifact_path: Path,
    dataset_cache: SignalDatasetCache | None = None,
) -> dict[str, object]:
    """运行完整 OOS 测试窗口。"""

    return _evaluate_signal_window(
        spec=spec,
        artifact_path=artifact_path,
        start_date=spec.signal_test.start_date,
        end_date=spec.signal_test.end_date,
        dataset_cache=dataset_cache,
        window_name=spec.signal_test.name or "signal_test",
    )


def _evaluate_signal_window(
    *,
    spec: ExperimentSpec,
    artifact_path: Path,
    start_date: str,
    end_date: str,
    dataset_cache: SignalDatasetCache | None = None,
    window_name: str | None = None,
) -> dict[str, object]:
    """评估一个指定时间窗口内的信号表现。"""

    estimator, metadata = load_signal_artifact(artifact_path)
    dataset = _build_signal_frame(
        spec=spec,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
        dataset_cache=dataset_cache,
    )
    return _evaluate_signal_dataset(
        estimator=estimator,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        window_name=window_name,
    )


def _build_signal_frame(
    *,
    spec: ExperimentSpec,
    metadata: dict[str, object],
    start_date: str,
    end_date: str,
    dataset_cache: SignalDatasetCache | None = None,
) -> PreparedSignalDataset:
    """为 signal_test / signal_windows 构建评估数据集。"""

    feature_columns = list(metadata["feature_columns"])
    history_padding_days = estimate_factor_history_lookback(feature_columns)
    return prepare_signal_dataset(
        market=str(metadata.get("market", spec.market)),
        provider_name=spec.provider,
        data_root=spec.data_root,
        universe_root=spec.universe_root,
        reference_root=str(metadata.get("reference_root", spec.reference_root)),
        adjust=spec.adjust,
        timeframe=spec.timeframe,
        symbols=spec.symbols or None,
        universe=spec.universe,
        start_date=start_date,
        end_date=end_date,
        feature_columns=feature_columns,
        feature_normalization=str(metadata.get("feature_normalization", spec.feature_normalization)),
        label_horizon=int(metadata["label_horizon"]),
        target_mode=str(metadata.get("target_mode", spec.train.target_mode)),
        progress_desc="Evaluating signal window",
        dataset_cache=dataset_cache,
        history_padding_days=history_padding_days,
        trim_to_requested_range=True,
    )


def _validate_experiment_windows(spec: ExperimentSpec) -> None:
    """检查实验关键时间窗是否满足 OOS 约束。"""

    if spec.signal_test.start_date <= spec.train.end_date:
        raise ValueError("signal_test.start_date must be later than train.end_date so final testing stays fully out-of-sample.")
    if spec.train.tuning and spec.train.valid_end_date and spec.train.valid_end_date != spec.train.end_date:
        raise ValueError("When train.tuning is configured, train.valid_end_date must equal train.end_date.")


def _evaluate_candidate_record(
    *,
    candidate_index: int,
    trial_record: dict[str, object],
    spec: ExperimentSpec,
    train_dataset: PreparedSignalDataset,
    selection_dataset: PreparedSignalDataset,
) -> dict[str, object]:
    """评估单个候选参数组合。"""

    params = dict(trial_record.get("params", {}))
    estimator = fit_model(
        frame=train_dataset.frame,
        feature_columns=train_dataset.feature_columns,
        label_column=train_dataset.label_column,
        model_name=spec.model,
        model_params=params,
    )
    selection_result = _evaluate_signal_dataset(
        estimator=estimator,
        dataset=selection_dataset,
        start_date=str(spec.train.valid_start_date),
        end_date=str(spec.train.valid_end_date),
    )
    return {
        "candidate_index": candidate_index,
        "trial_number": trial_record.get("trial_number"),
        "tuning_score": trial_record.get("score"),
        "model_params": params,
        "selection_signal_metrics": dict(selection_result["metrics"]),
        "selection_rows": int(selection_result["rows"]),
    }


def _evaluate_signal_dataset(
    *,
    estimator,
    dataset: PreparedSignalDataset,
    start_date: str,
    end_date: str,
    window_name: str | None = None,
) -> dict[str, object]:
    """对一个已经准备好的数据集直接计算信号指标。"""

    metrics = evaluate_model(
        estimator=estimator,
        frame=dataset.frame,
        feature_columns=dataset.feature_columns,
        label_column=dataset.label_column,
    )
    return {
        "name": window_name,
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(dataset.frame)),
        "metrics": metrics.as_dict(),
    }


def _run_signal_windows(
    *,
    spec: ExperimentSpec,
    artifact_path: Path,
    dataset_cache: SignalDatasetCache | None = None,
) -> list[dict[str, object]]:
    """逐个执行配置里的分段 OOS 窗口。"""

    windows = list(spec.signal_windows)
    return [
        _evaluate_signal_window(
            spec=spec,
            artifact_path=artifact_path,
            start_date=window.start_date,
            end_date=window.end_date,
            dataset_cache=dataset_cache,
            window_name=window.name or f"signal_window_{index}",
        )
        for index, window in enumerate(windows, start=1)
    ]


def _resolve_candidate_selection_workers(
    *,
    spec: ExperimentSpec,
    tuning_summary: dict[str, object],
    candidate_count: int,
) -> int:
    """决定 candidate selection 最多开多少并发。"""

    if candidate_count <= 1:
        return 1
    if model_uses_gpu(spec.model, dict(spec.model_params)):
        gpu_devices = [str(item) for item in tuning_summary.get("gpu_devices", []) if str(item).strip()]
        return max(1, min(candidate_count, len(gpu_devices) or 1))
    runtime = get_runtime_inventory()
    cpu_threads_per_trial = max(1, int(tuning_summary.get("cpu_threads_per_trial") or 1))
    return max(1, min(candidate_count, runtime.logical_cpu_count // cpu_threads_per_trial))


def _serialize_execution_options(options: GroupExecutionOptions) -> dict[str, object]:
    """把 group 执行参数落成可序列化的 summary 字段。"""

    return {
        "parallel": bool(options.parallel),
        "cpu_workers": options.cpu_workers,
        "gpu_workers": options.gpu_workers,
        "gpu_devices": list(options.gpu_devices or []),
        "cpu_threads_per_job": options.cpu_threads_per_job,
    }


def _track_group_task(
    *,
    tracker: GroupRunTracker,
    experiment_path: str,
    summary: dict[str, object] | None,
    error: str | None,
) -> None:
    """把单个实验结果写回组级 tracker。"""

    if summary is not None:
        tracker.record_completed(summary)
        return
    tracker.record_failure(experiment_path=experiment_path, error=error or "Unknown error")
