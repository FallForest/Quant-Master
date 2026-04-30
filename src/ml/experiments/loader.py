from __future__ import annotations

"""实验配置加载器。"""

from pathlib import Path

import yaml

from ml.experiments.specs import (
    ExperimentGroupSpec,
    ExperimentReportSpec,
    ExperimentSignalTestSpec,
    ExperimentSpec,
    ExperimentTrainSpec,
    ExperimentTuningSpec,
    ExperimentWalkForwardSpec,
)
from ml.selection import CandidateSelectionConfig
from research.profiles import apply_profile_defaults, load_research_profile


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """把单实验 YAML 解析成 `ExperimentSpec`。

    这里做的是“结构化读取 + 默认值补全”，不是完整业务校验。
    更严格的时间窗约束会在 runner 中继续检查。
    """

    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    profile = load_research_profile(raw["research_profile"]) if raw.get("research_profile") else None
    raw = apply_profile_defaults(raw, profile=profile)
    train_raw = raw.get("train") or {}
    signal_test_raw = raw.get("signal_test") or {}
    # 项目已经统一迁移到 signal_test 口径，不再接受旧 backtest 别名。
    if not signal_test_raw:
        raise ValueError("Experiment spec must define a `signal_test` block. Legacy `backtest` aliases are no longer supported.")
    signal_windows_raw = raw.get("signal_windows") or []
    report_raw = raw.get("report") or {}
    walk_forward_raw = train_raw.get("walk_forward") or None
    tuning_raw = train_raw.get("tuning") or None
    candidate_selection_raw = train_raw.get("candidate_selection") or {}
    return ExperimentSpec(
        name=str(raw["name"]),
        research_profile=str(raw["research_profile"]) if raw.get("research_profile") else None,
        market=str(raw.get("market", "ashare")),
        provider=str(raw.get("provider", "parquet")),
        data_root=str(raw.get("data_root", "data/lake")),
        universe_root=str(raw.get("universe_root", "data/universe")),
        reference_root=str(raw.get("reference_root", "data/reference")),
        timeframe=str(raw.get("timeframe", "1d")),
        adjust=str(raw.get("adjust", "qfq")),
        symbols=[str(item) for item in raw.get("symbols", [])],
        universe=str(raw["universe"]) if raw.get("universe") is not None else None,
        benchmark_symbol=str(raw.get("benchmark_symbol", "sh000300")),
        industry_standard=str(raw.get("industry_standard", "申银万国行业分类标准")),
        market_cap_bucket_count=int(raw.get("market_cap_bucket_count", 5)),
        baseline_manifest_path=(
            str(raw["baseline_manifest_path"])
            if raw.get("baseline_manifest_path") is not None
            else None
        ),
        features=[str(item) for item in raw.get("features", [])],
        feature_normalization=str(raw.get("feature_normalization", "none")),
        ic_decay_horizons=[int(item) for item in raw.get("ic_decay_horizons", [1, 5, 10, 20])],
        model=str(raw.get("model", "ridge")),
        model_params=dict(raw.get("model_params") or {}),
        train=ExperimentTrainSpec(
            start_date=str(train_raw["start_date"]),
            end_date=str(train_raw["end_date"]),
            validation_mode=str(train_raw.get("validation_mode", "holdout")),
            train_end_date=str(train_raw["train_end_date"]) if train_raw.get("train_end_date") else None,
            valid_start_date=str(train_raw["valid_start_date"]) if train_raw.get("valid_start_date") else None,
            valid_end_date=str(train_raw["valid_end_date"]) if train_raw.get("valid_end_date") else None,
            label_horizon=int(train_raw.get("label_horizon", 5)),
            target_mode=str(train_raw.get("target_mode", "future_return")),
            purge_size=int(train_raw.get("purge_size", 0)),
            embargo_size=int(train_raw.get("embargo_size", 0)),
            walk_forward=ExperimentWalkForwardSpec(
                train_size=int(walk_forward_raw["train_size"]),
                valid_size=int(walk_forward_raw["valid_size"]),
                step_size=int(walk_forward_raw["step_size"]) if walk_forward_raw.get("step_size") is not None else None,
                expanding=bool(walk_forward_raw.get("expanding", True)),
                purge_size=int(walk_forward_raw.get("purge_size", train_raw.get("purge_size", 0))),
                embargo_size=int(walk_forward_raw.get("embargo_size", train_raw.get("embargo_size", 0))),
            )
            if walk_forward_raw
            else None,
            tuning=ExperimentTuningSpec(
                trials=int(tuning_raw.get("trials", 20)),
                metric=str(tuning_raw.get("metric", "spearman_ic")),
                direction=str(tuning_raw.get("direction", "maximize")),
                timeout_seconds=int(tuning_raw["timeout_seconds"]) if tuning_raw.get("timeout_seconds") is not None else None,
                seed=int(tuning_raw.get("seed", 42)),
                keep_top_trials=int(tuning_raw.get("keep_top_trials", 5)),
                parallel_jobs=int(tuning_raw["parallel_jobs"]) if tuning_raw.get("parallel_jobs") is not None else None,
                gpu_devices=[str(item) for item in tuning_raw.get("gpu_devices", [])],
                cpu_threads_per_trial=(
                    int(tuning_raw["cpu_threads_per_trial"])
                    if tuning_raw.get("cpu_threads_per_trial") is not None
                    else None
                ),
            )
            if tuning_raw
            else None,
            candidate_selection=CandidateSelectionConfig(
                top_k=int(candidate_selection_raw.get("top_k", 5)),
                metric=str(candidate_selection_raw.get("metric", "oos_spearman_ic")),
                direction=str(candidate_selection_raw.get("direction", "maximize")),
            ),
        ),
        signal_test=ExperimentSignalTestSpec(
            name=str(signal_test_raw["name"]) if signal_test_raw.get("name") else None,
            start_date=str(signal_test_raw["start_date"]),
            end_date=str(signal_test_raw["end_date"]),
        ),
        signal_windows=[
            ExperimentSignalTestSpec(
                name=str(item["name"]) if item.get("name") else None,
                start_date=str(item["start_date"]),
                end_date=str(item["end_date"]),
            )
            for item in signal_windows_raw
        ],
        report=ExperimentReportSpec(
            output_dir=str(report_raw["output_dir"]) if report_raw.get("output_dir") else None,
            artifact_path=str(report_raw["artifact_path"]) if report_raw.get("artifact_path") else None,
        ),
        group=str(raw["group"]) if raw.get("group") else None,
    )


def load_experiment_group_spec(path: str | Path) -> ExperimentGroupSpec:
    """把实验组 YAML 解析成 `ExperimentGroupSpec`。"""

    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    profile = load_research_profile(raw["research_profile"]) if raw.get("research_profile") else None
    raw = apply_profile_defaults(raw, profile=profile)
    return ExperimentGroupSpec(
        name=str(raw["name"]),
        experiments=[str(item) for item in raw.get("experiments", [])],
        output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
        research_profile=str(raw["research_profile"]) if raw.get("research_profile") else None,
        baseline_manifest_path=(
            str(raw["baseline_manifest_path"])
            if raw.get("baseline_manifest_path") is not None
            else None
        ),
    )
