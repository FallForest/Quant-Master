from __future__ import annotations

"""实验组调度器。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from data.thread_parallel import ThreadTaskOutcome, run_bounded_thread_pool
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
    """实验组执行参数。

    参数说明：
    - parallel:
      是否启用并发调度。为 False 时按顺序串行执行。
    - cpu_workers:
      最多同时跑多少个 CPU 实验。这里的“实验”是 group 里的独立 YAML。
    - gpu_workers:
      最多同时跑多少个 GPU 实验。如果传了 gpu_devices，还会受设备数上限约束。
    - gpu_devices:
      允许调度器使用的 GPU 设备编号，例如 ["0", "1"]。
    - cpu_threads_per_job:
      每个实验内部最多使用多少 CPU 线程。会被写回具体模型参数，
      防止“组级并发”和“模型内多线程”叠加后过度抢占 CPU。
    """

    parallel: bool = False
    cpu_workers: int | None = None
    gpu_workers: int | None = None
    gpu_devices: list[str] | None = None
    cpu_threads_per_job: int | None = None


@dataclass(slots=True, frozen=True)
class ScheduledExperiment:
    """调度器内部使用的轻量任务描述。"""

    experiment_path: str
    uses_gpu: bool


def execute_group(
    *,
    experiment_paths: list[str],
    continue_on_error: bool,
    options: GroupExecutionOptions,
    on_task_complete: Callable[[str, dict[str, object] | None, str | None], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """执行一组实验。

    返回值：
    - summaries: 成功实验的 summary 列表
    - failures: 失败实验及错误信息
    """

    scheduled = [
        ScheduledExperiment(
            experiment_path=str(path),
            uses_gpu=model_uses_gpu(spec.model, spec.model_params),
        )
        for path, spec in ((path, load_experiment_spec(path)) for path in experiment_paths)
    ]
    # 先识别本机资源，再把 group 级配置解释成真正的 CPU/GPU 槽位数量。
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

    # 未开启并发、或任务数/资源数不足时，退化为串行链路，逻辑更简单也更稳。
    if not options.parallel or max_workers <= 1 or len(experiment_paths) <= 1:
        summaries, failures = _run_sequential(
            experiment_paths=experiment_paths,
            continue_on_error=continue_on_error,
            cpu_threads_per_job=cpu_threads_per_job,
            on_task_complete=on_task_complete,
        )
        return _sort_summaries(summaries, experiment_paths=experiment_paths), failures

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    pending = list(scheduled)
    available_cpu_slots = cpu_workers
    available_gpu_slots = list(gpu_slots)

    def _submit_task(item: ScheduledExperiment):
        # 调度器不直接区分“哪台 GPU 更好”，只负责把任务分配到空闲槽位。
        nonlocal available_cpu_slots
        if item.uses_gpu:
            if not available_gpu_slots:
                return None
            gpu_device = available_gpu_slots.pop(0)
            return (
                lambda: _run_experiment_task(
                    experiment_path=item.experiment_path,
                    cpu_threads_per_job=cpu_threads_per_job,
                    gpu_device=gpu_device,
                ),
                {"uses_gpu": True, "gpu_device": gpu_device},
            )
        if available_cpu_slots <= 0:
            return None
        available_cpu_slots -= 1
        return (
            lambda: _run_experiment_task(
                experiment_path=item.experiment_path,
                cpu_threads_per_job=cpu_threads_per_job,
                gpu_device=None,
            ),
            {"uses_gpu": False, "gpu_device": None},
        )

    def _handle_outcome(outcome: ThreadTaskOutcome[ScheduledExperiment, dict[str, object], dict[str, object]]) -> None:
        # 任务结束后先归还资源槽位，再记录 summary / failure。
        nonlocal available_cpu_slots
        context = outcome.context or {}
        if bool(context.get("uses_gpu")):
            gpu_device = context.get("gpu_device")
            if gpu_device is not None:
                available_gpu_slots.append(str(gpu_device))
        else:
            available_cpu_slots += 1
        if outcome.succeeded:
            assert outcome.value is not None
            summaries.append(outcome.value)
            if on_task_complete is not None:
                on_task_complete(outcome.item.experiment_path, outcome.value, None)
            return
        assert outcome.error is not None
        failures.append({"experiment_path": outcome.item.experiment_path, "error": str(outcome.error)})
        if on_task_complete is not None:
            on_task_complete(outcome.item.experiment_path, None, str(outcome.error))

    run_bounded_thread_pool(
        items=pending,
        max_workers=max_workers,
        submitter=_submit_task,
        on_complete=_handle_outcome,
        continue_on_error=continue_on_error,
    )
    return _sort_summaries(summaries, experiment_paths=experiment_paths), failures


def _run_sequential(
    *,
    experiment_paths: list[str],
    continue_on_error: bool,
    cpu_threads_per_job: int,
    on_task_complete: Callable[[str, dict[str, object] | None, str | None], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """串行执行 group 中的实验。"""

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for experiment_path in experiment_paths:
        try:
            summary = _run_experiment_task(
                experiment_path=experiment_path,
                cpu_threads_per_job=cpu_threads_per_job,
                gpu_device=None,
            )
            summaries.append(summary)
            if on_task_complete is not None:
                on_task_complete(experiment_path, summary, None)
        except Exception as exc:  # noqa: BLE001
            failures.append({"experiment_path": str(experiment_path), "error": str(exc)})
            if on_task_complete is not None:
                on_task_complete(str(experiment_path), None, str(exc))
            if not continue_on_error:
                raise
    return summaries, failures


def _run_experiment_task(
    *,
    experiment_path: str,
    cpu_threads_per_job: int,
    gpu_device: str | None,
) -> dict[str, object]:
    """单个调度任务的真正执行体。"""

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
    """把组级运行时约束写回到具体实验配置。"""

    spec.model_params = apply_runtime_model_params(
        model_name=spec.model,
        model_params=spec.model_params,
        cpu_threads=cpu_threads_per_job,
        gpu_device=gpu_device,
    )
    if spec.train.tuning is not None:
        spec.train.tuning.parallel_jobs = 1 if spec.train.tuning.parallel_jobs is None else spec.train.tuning.parallel_jobs
        spec.train.tuning.gpu_devices = [gpu_device] if gpu_device is not None else list(spec.train.tuning.gpu_devices)
        spec.train.tuning.cpu_threads_per_trial = cpu_threads_per_job
    return spec


def _resolve_cpu_workers(*, requested: int | None, experiment_count: int) -> int:
    """把请求的 CPU worker 数裁剪到合理范围。"""

    if experiment_count <= 0:
        return 0
    if requested is None:
        return 1
    return max(0, min(int(requested), experiment_count))


def _resolve_gpu_slots(*, requested: int | None, gpu_devices: list[str], experiment_count: int) -> list[str | None]:
    """解析可用 GPU 槽位。"""

    if experiment_count <= 0:
        return []
    if gpu_devices:
        max_slots = len(gpu_devices)
        slot_count = max(1, min(requested if requested is not None else max_slots, max_slots, experiment_count))
        return gpu_devices[:slot_count]
    slot_count = max(1, min(int(requested) if requested is not None else 1, experiment_count))
    return [None] * slot_count


def _sort_summaries(summaries: list[dict[str, object]], *, experiment_paths: list[str]) -> list[dict[str, object]]:
    """按 group 配置里的实验顺序输出 summary，便于对照原始清单。"""

    order = {str(path): index for index, path in enumerate(experiment_paths)}
    return sorted(
        summaries,
        key=lambda item: order.get(str(item.get("experiment_path")), len(order)),
    )
