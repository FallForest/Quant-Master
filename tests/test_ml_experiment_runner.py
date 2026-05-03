from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.loader import load_experiment_spec
from ml.experiments.runner import (
    _resolve_candidate_selection_workers,
    _run_candidate_selection,
    run_experiment,
    run_experiment_group,
)
from ml.experiments.specs import ExperimentGroupSpec
from ml.models import ValidationMetrics
from ml.prepared_data import PreparedSignalDataset, SignalDatasetCache
from ml.runtime import RuntimeInventory


def test_run_candidate_selection_reuses_prepared_datasets(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "ridge.yaml"
    experiment.write_text(
        "\n".join(
            [
                "name: ridge_selection_demo",
                "model: ridge",
                "features: ['return_5']",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-06-30",
                "  valid_start_date: 2021-05-01",
                "  valid_end_date: 2021-06-30",
                "  tuning:",
                "    trials: 3",
                "signal_test:",
                "  start_date: 2021-07-01",
                "  end_date: 2021-07-31",
            ]
        ),
        encoding="utf-8",
    )
    spec = load_experiment_spec(experiment)

    prepared_calls: list[tuple[str, str]] = []
    train_dataset = PreparedSignalDataset(
        frame=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-03-01", "2021-03-02"]),
                "symbol": ["000001", "000002"],
                "return_5": [0.1, 0.2],
                "future_return_5": [0.2, 0.3],
            }
        ),
        feature_columns=["return_5"],
        label_column="future_return_5",
        symbols=["000001", "000002"],
    )
    selection_dataset = PreparedSignalDataset(
        frame=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-05-03", "2021-05-04"]),
                "symbol": ["000001", "000002"],
                "return_5": [0.3, 0.4],
                "future_return_5": [0.4, 0.5],
            }
        ),
        feature_columns=["return_5"],
        label_column="future_return_5",
        symbols=["000001", "000002"],
    )

    def fake_prepare_signal_dataset(**kwargs):
        prepared_calls.append((kwargs["start_date"], kwargs["end_date"]))
        if kwargs["end_date"] == "2021-04-30":
            return train_dataset
        return selection_dataset

    def fake_fit_model(**kwargs):
        return dict(kwargs["model_params"])

    def fake_evaluate_model(*, estimator, frame, feature_columns, label_column):
        score = float(estimator["selection_score"])
        return ValidationMetrics(
            mae=0.0,
            r2=0.0,
            ic=score,
            rank_ic=score,
            ic_std=1.0,
            rank_ic_std=1.0,
            icir=score,
            rank_icir=score,
            ndcg_at_10=score,
        )

    monkeypatch.setattr("ml.experiments.runner.prepare_signal_dataset", fake_prepare_signal_dataset)
    monkeypatch.setattr("ml.experiments.runner.fit_model", fake_fit_model)
    monkeypatch.setattr("ml.experiments.runner.evaluate_model", fake_evaluate_model)

    result = _run_candidate_selection(
        spec=spec,
        training_metadata={
            "tuning": {
                "trial_records": [
                    {"trial_number": 0, "score": 0.1, "params": {"selection_score": 0.1}},
                    {"trial_number": 1, "score": 0.3, "params": {"selection_score": 0.3}},
                    {"trial_number": 2, "score": 0.2, "params": {"selection_score": 0.2}},
                ],
                "cpu_threads_per_trial": 1,
                "gpu_devices": [],
            }
        },
        dataset_cache=SignalDatasetCache(),
    )

    assert prepared_calls == [("2021-01-01", "2021-04-30"), ("2021-05-01", "2021-06-30")]
    assert result["selected_model_params"]["selection_score"] == 0.3
    assert len(result["candidate_selection"]["candidates"]) == 3


def test_resolve_candidate_selection_workers_respects_cpu_and_gpu_capacity(tmp_path: Path, monkeypatch) -> None:
    cpu_experiment = tmp_path / "cpu.yaml"
    cpu_experiment.write_text(
        "\n".join(
            [
                "name: cpu_demo",
                "model: ridge",
                "features: ['return_5']",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-02-01",
                "signal_test:",
                "  start_date: 2021-03-01",
                "  end_date: 2021-03-31",
            ]
        ),
        encoding="utf-8",
    )
    gpu_experiment = tmp_path / "gpu.yaml"
    gpu_experiment.write_text(
        "\n".join(
            [
                "name: gpu_demo",
                "model: xgboost",
                "model_params:",
                "  device: cuda",
                "features: ['return_5']",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-02-01",
                "signal_test:",
                "  start_date: 2021-03-01",
                "  end_date: 2021-03-31",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ml.experiments.runner.get_runtime_inventory",
        lambda: RuntimeInventory(logical_cpu_count=16, gpu_devices=tuple()),
    )

    assert _resolve_candidate_selection_workers(
        spec=load_experiment_spec(cpu_experiment),
        tuning_summary={"cpu_threads_per_trial": 4, "gpu_devices": []},
        candidate_count=5,
    ) == 4
    assert _resolve_candidate_selection_workers(
        spec=load_experiment_spec(gpu_experiment),
        tuning_summary={"cpu_threads_per_trial": 8, "gpu_devices": ["0", "1"]},
        candidate_count=5,
    ) == 2


def test_run_experiment_final_retrain_uses_full_training_cache(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "full_retrain.yaml"
    report_dir = tmp_path / "report"
    artifact_dir = tmp_path / "artifact"
    experiment.write_text(
        "\n".join(
            [
                "name: full_retrain_demo",
                "model: ridge",
                "features: ['return_5']",
                "train:",
                "  start_date: 2021-01-01",
                "  end_date: 2021-06-30",
                "  valid_start_date: 2021-05-01",
                "  valid_end_date: 2021-06-30",
                "  tuning:",
                "    trials: 3",
                "signal_test:",
                "  start_date: 2021-07-01",
                "  end_date: 2021-07-31",
                "report:",
                f"  output_dir: {report_dir.as_posix()}",
                f"  artifact_path: {artifact_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    spec = load_experiment_spec(experiment)

    full_dataset = PreparedSignalDataset(
        frame=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-01-04", "2021-06-30"]),
                "symbol": ["000001", "000001"],
                "return_5": [0.1, 0.2],
                "future_return_5": [0.2, 0.3],
            }
        ),
        feature_columns=["return_5"],
        label_column="future_return_5",
        symbols=["000001"],
    )
    selection_dataset = PreparedSignalDataset(
        frame=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-01-04", "2021-04-30"]),
                "symbol": ["000001", "000001"],
                "return_5": [0.1, 0.2],
                "future_return_5": [0.2, 0.3],
            }
        ),
        feature_columns=["return_5"],
        label_column="future_return_5",
        symbols=["000001"],
    )

    def fake_prepare_signal_dataset(**kwargs):
        if kwargs["end_date"] == "2021-06-30":
            return full_dataset
        return selection_dataset

    train_calls: list[PreparedSignalDataset | None] = []

    def fake_train_ml_signal_model(**kwargs):
        train_calls.append(kwargs.get("prepared_dataset"))
        return {
            "model_params": dict(kwargs.get("model_params") or {"alpha": 1.0}),
            "feature_columns": ["return_5"],
            "label_column": "future_return_5",
            "validation_metrics": {"rank_ic": 0.1},
            "tuning": {"trial_records": []},
            "candidate_selection": {},
        }

    class FakeDiagnostics:
        def as_dict(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr("ml.experiments.runner.prepare_signal_dataset", fake_prepare_signal_dataset)
    monkeypatch.setattr("ml.experiments.runner.train_ml_signal_model", fake_train_ml_signal_model)
    monkeypatch.setattr(
        "ml.experiments.runner._run_candidate_selection",
        lambda **kwargs: {
            "selected_model_params": {"alpha": 2.0},
            "candidate_selection": {"enabled": True, "candidates": [], "selected_metric": "rank_ic"},
            "tuning": {"trial_records": []},
        },
    )
    monkeypatch.setattr(
        "ml.experiments.runner._run_signal_test",
        lambda **kwargs: {"metrics": {"rank_ic": 0.2, "rank_icir": 0.2}, "rows": 10},
    )
    monkeypatch.setattr(
        "ml.experiments.runner._run_signal_windows",
        lambda **kwargs: [{"name": "signal_test", "metrics": {"rank_ic": 0.2, "rank_icir": 0.2}, "rows": 10}],
    )
    monkeypatch.setattr(
        "ml.experiments.runner._build_signal_frame",
        lambda **kwargs: full_dataset,
    )
    monkeypatch.setattr(
        "ml.experiments.runner.load_signal_artifact",
        lambda *args, **kwargs: (object(), {"feature_columns": ["return_5"], "label_column": "future_return_5"}),
    )
    monkeypatch.setattr(
        "ml.experiments.runner.build_ic_decay_profile",
        lambda **kwargs: [{"horizon": 5, "spearman_ic": 0.2}],
    )
    monkeypatch.setattr(
        "ml.experiments.runner.build_signal_slice_diagnostics",
        lambda **kwargs: {"year_windows": [], "market_style_regimes": [], "industry_buckets": [], "market_cap_buckets": [], "meta": {}},
    )
    monkeypatch.setattr("ml.experiments.runner._rewrite_artifact_metadata", lambda **kwargs: None)
    monkeypatch.setattr("ml.experiments.runner.build_overfit_diagnostics", lambda **kwargs: FakeDiagnostics())

    run_experiment(spec=spec, experiment_path=experiment)

    assert train_calls[0] is None
    assert train_calls[1] is full_dataset


def test_run_experiment_group_resume_skips_completed_and_writes_manifests(tmp_path: Path, monkeypatch) -> None:
    group_output_dir = tmp_path / "group"
    experiment_a = str((tmp_path / "a.yaml").resolve())
    experiment_b = str((tmp_path / "b.yaml").resolve())
    existing_summary = {
        "name": "demo_group",
        "experiment_paths": [experiment_a, experiment_b],
        "experiment_count": 2,
        "completed_count": 1,
        "failed_count": 0,
        "experiments": [
            {
                "name": "exp_a",
                "experiment_path": experiment_a,
                "report_dir": str(tmp_path / "reports" / "a"),
                "artifact_path": str(tmp_path / "artifacts" / "a"),
            }
        ],
        "failures": [],
    }
    group_output_dir.mkdir(parents=True)
    (group_output_dir / "group_summary.json").write_text(json.dumps(existing_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    scheduled_paths: list[list[str]] = []

    def fake_execute_group(*, experiment_paths, continue_on_error, options, on_task_complete=None):
        scheduled_paths.append(list(experiment_paths))
        summary = {
            "name": "exp_b",
            "experiment_path": experiment_b,
            "report_dir": str(tmp_path / "reports" / "b"),
            "artifact_path": str(tmp_path / "artifacts" / "b"),
        }
        if on_task_complete is not None:
            on_task_complete(experiment_b, summary, None)
        return [summary], []

    monkeypatch.setattr("ml.experiments.runner.execute_group", fake_execute_group)

    summary = run_experiment_group(
        group_spec=ExperimentGroupSpec(name="demo_group", experiments=["a.yaml", "b.yaml"], output_dir=str(group_output_dir)),
        experiment_paths=[experiment_a, experiment_b],
        continue_on_error=True,
        resume=True,
    )

    manifest = json.loads((group_output_dir / "group_run_manifest.json").read_text(encoding="utf-8"))
    failures = json.loads((group_output_dir / "group_run_failures.json").read_text(encoding="utf-8"))

    assert scheduled_paths == [[experiment_b]]
    assert summary["completed_count"] == 2
    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 0
    assert manifest["resume"] is True
    assert failures["failure_count"] == 0
