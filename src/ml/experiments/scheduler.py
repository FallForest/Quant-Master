from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from ml.experiments.loader import load_experiment_spec
from ml.experiments.specs import ExperimentSpec
from ml.runtime import (
    apply_runtime_model_params,
    get_runtime_inventory,
    model_uses_gpu,
    resolve_cpu_threads_per_job,
)


@dataclass(slots=True, frozen=True)
class GroupExecutionOptions:
    parallel: bool = False
    cpu_workers: int | None = None
    gpu_workers: int | None = None
    gpu_devices: list[str] | None = None
    cpu_threads_per_job: int | None = None


@dataclass(slots=True, frozen=True)
class ScheduledExperiment:
    experiment_path: str
    uses_gpu: bool


@dataclass(slots=True)
class RunningTask:
    future: Future
    experiment_path: str
    uses_gpu: bool
    gpu_device: str | None


def execute_group(
    *,
    experiment_paths: list[str],
    continue_on_error: bool,
    options: GroupExecutionOptions,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    scheduled = [
        ScheduledExperiment(
            experiment_path=str(path),
            uses_gpu=model_uses_gpu(spec.model, spec.model_params),
        )
        for path, spec in ((path, load_experiment_spec(path)) for path in experiment_paths)
    ]
    runtime = get_runtime_inventory(preferred_gpu_devices=options.gpu_devices or None)
    cpu_workers = _resolve_cpu_workers(
        requested=options.cpu_workers,
        experiment_count=sum(1 for item in scheduled if not item.uses_gpu),
    )
    gpu_slots = _resolve_gpu_slots(
        requested=options.gpu_workers,
        gpu_devices=list(runtime.gpu_devices),
        experiment_count=sum(1 for item in scheduled if item.uses_gpu),
    )
    max_workers = max(1, cpu_workers + max(0, len(gpu_slots)))
    cpu_threads_per_job = options.cpu_threads_per_job or resolve_cpu_threads_per_job(
        logical_cpu_count=runtime.logical_cpu_count,
        concurrent_jobs=max_workers,
    )

    if not options.parallel or max_workers <= 1 or len(experiment_paths) <= 1:
        summaries, failures = _run_sequential(
            experiment_paths=experiment_paths,
            continue_on_error=continue_on_error,
            cpu_threads_per_job=cpu_threads_per_job,
        )
        return _sort_summaries(summaries, experiment_paths=experiment_paths), failures

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    pending = list(scheduled)
    available_cpu_slots = cpu_workers
    available_gpu_slots = list(gpu_slots)
    running: dict[Future, RunningTask] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            submitted = False
            index = 0
            while index < len(pending):
                item = pending[index]
                if item.uses_gpu:
                    if not available_gpu_slots:
                        index += 1
                        continue
                    gpu_device = available_gpu_slots.pop(0)
                    future = executor.submit(
                        _run_experiment_task,
                        experiment_path=item.experiment_path,
                        cpu_threads_per_job=cpu_threads_per_job,
                        gpu_device=gpu_device,
                    )
                    running[future] = RunningTask(
                        future=future,
                        experiment_path=item.experiment_path,
                        uses_gpu=True,
                        gpu_device=gpu_device,
                    )
                else:
                    if available_cpu_slots <= 0:
                        index += 1
                        continue
                    available_cpu_slots -= 1
                    future = executor.submit(
                        _run_experiment_task,
                        experiment_path=item.experiment_path,
                        cpu_threads_per_job=cpu_threads_per_job,
                        gpu_device=None,
                    )
                    running[future] = RunningTask(
                        future=future,
                        experiment_path=item.experiment_path,
                        uses_gpu=False,
                        gpu_device=None,
                    )
                pending.pop(index)
                submitted = True

            if not running:
                break

            if not submitted:
                completed, _ = wait(tuple(running.keys()), return_when=FIRST_COMPLETED)
            else:
                completed = {future for future in running if future.done()}
                if not completed:
                    completed, _ = wait(tuple(running.keys()), return_when=FIRST_COMPLETED)

            for future in completed:
                task = running.pop(future)
                if task.uses_gpu:
                    if task.gpu_device is not None:
                        available_gpu_slots.append(task.gpu_device)
                else:
                    available_cpu_slots += 1

                try:
                    summaries.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({"experiment_path": task.experiment_path, "error": str(exc)})
                    if not continue_on_error:
                        raise

    return _sort_summaries(summaries, experiment_paths=experiment_paths), failures


def _run_sequential(
    *,
    experiment_paths: list[str],
    continue_on_error: bool,
    cpu_threads_per_job: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for experiment_path in experiment_paths:
        try:
            summaries.append(
                _run_experiment_task(
                    experiment_path=experiment_path,
                    cpu_threads_per_job=cpu_threads_per_job,
                    gpu_device=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"experiment_path": str(experiment_path), "error": str(exc)})
            if not continue_on_error:
                raise
    return summaries, failures


def _run_experiment_task(
    *,
    experiment_path: str,
    cpu_threads_per_job: int,
    gpu_device: str | None,
) -> dict[str, object]:
    from ml.experiments.runner import run_experiment

    spec = load_experiment_spec(experiment_path)
    spec = _apply_runtime_overrides(
        spec=spec,
        cpu_threads_per_job=cpu_threads_per_job,
        gpu_device=gpu_device,
    )
    return run_experiment(spec=spec, experiment_path=Path(experiment_path))


def _apply_runtime_overrides(
    *,
    spec: ExperimentSpec,
    cpu_threads_per_job: int,
    gpu_device: str | None,
) -> ExperimentSpec:
    spec.model_params = apply_runtime_model_params(
        model_name=spec.model,
        model_params=spec.model_params,
        cpu_threads=cpu_threads_per_job,
        gpu_device=gpu_device,
    )
    if spec.train.tuning is not None:
        spec.train.tuning.parallel_jobs = 1 if spec.train.tuning.parallel_jobs is None else spec.train.tuning.parallel_jobs
        spec.train.tuning.gpu_devices = [gpu_device] if gpu_device is not None else list(spec.train.tuning.gpu_devices)
    return spec


def _resolve_cpu_workers(*, requested: int | None, experiment_count: int) -> int:
    if experiment_count <= 0:
        return 0
    if requested is None:
        return 1
    return max(0, min(int(requested), experiment_count))


def _resolve_gpu_slots(*, requested: int | None, gpu_devices: list[str], experiment_count: int) -> list[str | None]:
    if experiment_count <= 0:
        return []
    if gpu_devices:
        max_slots = len(gpu_devices)
        slot_count = max(1, min(requested if requested is not None else max_slots, max_slots, experiment_count))
        return gpu_devices[:slot_count]
    slot_count = max(1, min(int(requested) if requested is not None else 1, experiment_count))
    return [None] * slot_count


def _sort_summaries(summaries: list[dict[str, object]], *, experiment_paths: list[str]) -> list[dict[str, object]]:
    order = {str(path): index for index, path in enumerate(experiment_paths)}
    return sorted(
        summaries,
        key=lambda item: order.get(str(item.get("experiment_path")), len(order)),
    )
