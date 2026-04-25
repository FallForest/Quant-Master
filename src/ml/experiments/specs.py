from __future__ import annotations

from dataclasses import dataclass, field

from ml.selection import CandidateSelectionConfig


@dataclass(slots=True)
class ExperimentWalkForwardSpec:
    train_size: int
    valid_size: int
    step_size: int | None = None
    expanding: bool = True
    purge_size: int = 0
    embargo_size: int = 0


@dataclass(slots=True)
class ExperimentTuningSpec:
    trials: int = 20
    metric: str = "spearman_ic"
    direction: str = "maximize"
    timeout_seconds: int | None = None
    seed: int = 42
    keep_top_trials: int = 5
    parallel_jobs: int | None = None
    gpu_devices: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentTrainSpec:
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
    start_date: str
    end_date: str


@dataclass(slots=True)
class ExperimentReportSpec:
    output_dir: str | None = None
    artifact_path: str | None = None


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    market: str = "ashare"
    provider: str = "parquet"
    data_root: str = "data/lake"
    universe_root: str = "data/universe"
    reference_root: str = "data/reference"
    timeframe: str = "1d"
    adjust: str = "qfq"
    symbols: list[str] = field(default_factory=list)
    universe: str | None = None
    features: list[str] = field(default_factory=list)
    model: str = "ridge"
    model_params: dict[str, object] = field(default_factory=dict)
    train: ExperimentTrainSpec = field(default_factory=lambda: ExperimentTrainSpec(start_date="2024-01-01", end_date="2024-12-31"))
    signal_test: ExperimentSignalTestSpec = field(
        default_factory=lambda: ExperimentSignalTestSpec(start_date="2025-01-01", end_date="2025-12-31")
    )
    report: ExperimentReportSpec = field(default_factory=ExperimentReportSpec)
    group: str | None = None


@dataclass(slots=True)
class ExperimentGroupSpec:
    name: str
    experiments: list[str]
    output_dir: str | None = None
