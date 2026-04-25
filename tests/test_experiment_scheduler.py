from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.scheduler import GroupExecutionOptions, execute_group


def test_execute_group_parallel_preserves_input_order(tmp_path: Path, monkeypatch) -> None:
    cpu_experiment = tmp_path / "cpu.yaml"
    gpu_experiment = tmp_path / "gpu.yaml"
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

    def fake_run(*, experiment_path: str, cpu_threads_per_job: int, gpu_device: str | None) -> dict[str, object]:
        if experiment_path == str(cpu_experiment):
            time.sleep(0.05)
        return {
            "name": Path(experiment_path).stem,
            "experiment_path": experiment_path,
            "cpu_threads_per_job": cpu_threads_per_job,
            "gpu_device": gpu_device,
        }

    monkeypatch.setattr("ml.experiments.scheduler._run_experiment_task", fake_run)
    summaries, failures = execute_group(
        experiment_paths=[str(cpu_experiment), str(gpu_experiment)],
        continue_on_error=False,
        options=GroupExecutionOptions(
            parallel=True,
            cpu_workers=1,
            gpu_workers=1,
            gpu_devices=["0"],
        ),
    )

    assert failures == []
    assert [item["experiment_path"] for item in summaries] == [str(cpu_experiment), str(gpu_experiment)]
    assert summaries[1]["gpu_device"] == "0"
