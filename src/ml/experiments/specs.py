from __future__ import annotations

"""实验配置对象定义。

这些 dataclass 是 YAML 配置在内存中的强类型表示。
loader 负责把 YAML 解析成这些对象，runner / training / backfill 再基于它们工作。
"""

from dataclasses import dataclass, field

from ml.selection import CandidateSelectionConfig


@dataclass(slots=True)
class ExperimentWalkForwardSpec:
    """滚动验证配置。

    `train_size` / `valid_size` / `step_size` 都按“交易日数量”解释，而不是自然日。
    """

    train_size: int
    valid_size: int
    step_size: int | None = None
    expanding: bool = True
    purge_size: int = 0
    embargo_size: int = 0


@dataclass(slots=True)
class ExperimentTuningSpec:
    """超参搜索配置。

    这里控制的是“候选参数如何产生”，不是最终模型如何评估。
    最终采用哪组参数，还要经过 candidate selection 这层筛选。
    """

    trials: int = 20
    metric: str = "spearman_ic"
    direction: str = "maximize"
    timeout_seconds: int | None = None
    seed: int = 42
    keep_top_trials: int = 5
    parallel_jobs: int | None = None
    gpu_devices: list[str] = field(default_factory=list)
    cpu_threads_per_trial: int | None = None


@dataclass(slots=True)
class ExperimentTrainSpec:
    """训练阶段配置。

    定义：
    1. 训练数据时间范围
    2. 标签构造方式
    3. 验证方式（holdout / walk_forward）
    4. 是否启用 tuning 与 candidate selection
    """

    start_date: str
    end_date: str
    validation_mode: str = "holdout"
    train_end_date: str | None = None
    valid_start_date: str | None = None
    valid_end_date: str | None = None
    label_horizon: int = 5
    target_mode: str = "future_return"
    purge_size: int = 0
    embargo_size: int = 0
    walk_forward: ExperimentWalkForwardSpec | None = None
    tuning: ExperimentTuningSpec | None = None
    candidate_selection: CandidateSelectionConfig = field(default_factory=CandidateSelectionConfig)


@dataclass(slots=True)
class ExperimentSignalTestSpec:
    """OOS 信号评估窗口。

    既可以表示完整的 `signal_test`，也可以表示拆分后的分段窗口。
    """

    start_date: str
    end_date: str
    name: str | None = None


@dataclass(slots=True)
class ExperimentReportSpec:
    """实验产物输出位置。"""

    output_dir: str | None = None
    artifact_path: str | None = None


@dataclass(slots=True)
class ExperimentSpec:
    """单个实验的完整配置对象。"""

    name: str
    research_profile: str | None = None
    market: str = "ashare"
    provider: str = "parquet"
    data_root: str = "data/lake"
    universe_root: str = "data/universe"
    reference_root: str = "data/reference"
    timeframe: str = "1d"
    adjust: str = "qfq"
    symbols: list[str] = field(default_factory=list)
    universe: str | None = None
    benchmark_symbol: str = "sh000300"
    industry_standard: str = "申银万国行业分类标准"
    market_cap_bucket_count: int = 5
    baseline_manifest_path: str | None = None
    features: list[str] = field(default_factory=list)
    feature_normalization: str = "none"
    ic_decay_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    model: str = "ridge"
    model_params: dict[str, object] = field(default_factory=dict)
    train: ExperimentTrainSpec = field(default_factory=lambda: ExperimentTrainSpec(start_date="2024-01-01", end_date="2024-12-31"))
    signal_test: ExperimentSignalTestSpec = field(
        default_factory=lambda: ExperimentSignalTestSpec(start_date="2025-01-01", end_date="2025-12-31")
    )
    signal_windows: list[ExperimentSignalTestSpec] = field(default_factory=list)
    report: ExperimentReportSpec = field(default_factory=ExperimentReportSpec)
    group: str | None = None


@dataclass(slots=True)
class ExperimentGroupSpec:
    """实验组配置。

    实验组只负责收集若干实验路径，并指定组级汇总输出目录。
    """

    name: str
    experiments: list[str]
    output_dir: str | None = None
    research_profile: str | None = None
    baseline_manifest_path: str | None = None
