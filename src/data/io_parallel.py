from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
import sys
from typing import Callable, Generic, Sequence, TypeVar

from tqdm import tqdm

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional fallback during partial installs
    psutil = None


TItem = TypeVar("TItem")
TResult = TypeVar("TResult")

DEFAULT_IO_CPU_TARGET_PCT = 72.0
DEFAULT_IO_BATCH_SIZE_MULTIPLIER = 2
DEFAULT_IO_AUTO_WORKER_MULTIPLIER = 4
DEFAULT_IO_AUTO_WORKER_CAP = 32


@dataclass(frozen=True)
class CpuUsageSnapshot:
    overall_pct: float
    average_core_pct: float
    peak_core_pct: float
    active_core_count: int
    core_count: int


@dataclass(frozen=True)
class AdaptiveIoBatchTelemetry:
    batch_index: int
    workers: int
    batch_size: int
    pending_after_batch: int
    cpu: CpuUsageSnapshot


@dataclass(frozen=True)
class AdaptiveIoOutcome(Generic[TItem, TResult]):
    item: TItem
    value: TResult | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class AdaptiveIoReport(Generic[TItem, TResult]):
    outcomes: list[AdaptiveIoOutcome[TItem, TResult]]
    telemetry: list[AdaptiveIoBatchTelemetry]
    planned_max_workers: int
    initial_workers: int

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.succeeded)


def resolve_auto_io_workers(max_workers: int | None = None) -> int:
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1.")
        return int(max_workers)
    logical_cores = cpu_count() or 4
    return max(4, min(DEFAULT_IO_AUTO_WORKER_CAP, logical_cores * DEFAULT_IO_AUTO_WORKER_MULTIPLIER))


def run_adaptive_io_tasks(
    *,
    items: Sequence[TItem],
    worker_fn: Callable[[TItem], TResult],
    progress_desc: str,
    progress_unit: str = "item",
    show_progress: bool = True,
    max_workers: int | None = None,
    min_workers: int = 1,
    cpu_target_pct: float = DEFAULT_IO_CPU_TARGET_PCT,
    batch_size_multiplier: int = DEFAULT_IO_BATCH_SIZE_MULTIPLIER,
    on_outcome: Callable[[AdaptiveIoOutcome[TItem, TResult]], None] | None = None,
    progress_factory=tqdm,
) -> AdaptiveIoReport[TItem, TResult]:
    item_list = list(items)
    if not item_list:
        return AdaptiveIoReport(outcomes=[], telemetry=[], planned_max_workers=1, initial_workers=1)

    planned_max_workers = min(resolve_auto_io_workers(max_workers), len(item_list))
    normalized_min_workers = min(max(1, min_workers), planned_max_workers)
    initial_workers = _resolve_initial_workers(
        planned_max_workers=planned_max_workers,
        normalized_min_workers=normalized_min_workers,
    )
    current_workers = initial_workers
    controller = _AdaptiveIoController(
        min_workers=normalized_min_workers,
        max_workers=planned_max_workers,
        cpu_target_pct=cpu_target_pct,
    )
    controller.prime()

    outcomes: list[AdaptiveIoOutcome[TItem, TResult] | None] = [None] * len(item_list)
    telemetry: list[AdaptiveIoBatchTelemetry] = []
    next_index = 0
    batch_index = 0
    progress = progress_factory(
        total=len(item_list),
        desc=f"{progress_desc} (adaptive <= {planned_max_workers} threads)",
        unit=progress_unit,
        file=sys.stdout,
        disable=not show_progress,
    )
    with progress as progress_bar:
        _update_progress_postfix(
            progress_bar,
            workers=current_workers,
            pending=len(item_list),
            cpu=CpuUsageSnapshot(0.0, 0.0, 0.0, 0, controller.core_count),
        )
        while next_index < len(item_list):
            batch_size = min(
                len(item_list) - next_index,
                max(current_workers, current_workers * max(1, batch_size_multiplier)),
            )
            indexed_batch = list(enumerate(item_list[next_index : next_index + batch_size], start=next_index))

            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                future_to_index = {
                    executor.submit(worker_fn, item): (index, item)
                    for index, item in indexed_batch
                }
                for future in as_completed(future_to_index):
                    index, item = future_to_index[future]
                    try:
                        outcomes[index] = AdaptiveIoOutcome(item=item, value=future.result())
                    except Exception as exc:  # noqa: BLE001
                        outcomes[index] = AdaptiveIoOutcome(item=item, error=exc)
                    finally:
                        if on_outcome is not None:
                            assert outcomes[index] is not None
                            on_outcome(outcomes[index])
                        progress_bar.update(1)

            next_index += batch_size
            snapshot = controller.snapshot()
            telemetry.append(
                AdaptiveIoBatchTelemetry(
                    batch_index=batch_index,
                    workers=current_workers,
                    batch_size=batch_size,
                    pending_after_batch=len(item_list) - next_index,
                    cpu=snapshot,
                )
            )
            current_workers = controller.next_workers(current_workers=current_workers, cpu=snapshot)
            _update_progress_postfix(
                progress_bar,
                workers=current_workers,
                pending=len(item_list) - next_index,
                cpu=snapshot,
            )
            batch_index += 1

    finalized_outcomes = [outcome for outcome in outcomes if outcome is not None]
    return AdaptiveIoReport(
        outcomes=finalized_outcomes,
        telemetry=telemetry,
        planned_max_workers=planned_max_workers,
        initial_workers=initial_workers,
    )


def format_adaptive_io_failures(
    failures: Sequence[AdaptiveIoOutcome[TItem, TResult]],
    *,
    max_examples: int = 5,
) -> str:
    return ", ".join(
        f"{outcome.item} ({type(outcome.error).__name__}: {outcome.error})"
        for outcome in failures[:max_examples]
        if outcome.error is not None
    )


class _AdaptiveIoController:
    def __init__(self, *, min_workers: int, max_workers: int, cpu_target_pct: float) -> None:
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cpu_target_pct = float(cpu_target_pct)
        self.core_count = _resolve_core_count()

    def prime(self) -> None:
        _sample_cpu_usage()

    def snapshot(self) -> CpuUsageSnapshot:
        return _sample_cpu_usage()

    def next_workers(self, *, current_workers: int, cpu: CpuUsageSnapshot) -> int:
        if self.max_workers <= self.min_workers:
            return self.max_workers

        hard_limit = max(self.cpu_target_pct + 15.0, 90.0)
        low_limit = max(10.0, self.cpu_target_pct - 20.0)
        scale_step = max(1, current_workers // 3)

        if cpu.overall_pct >= hard_limit or cpu.peak_core_pct >= 98.0:
            return max(self.min_workers, current_workers - scale_step)
        if cpu.overall_pct >= self.cpu_target_pct or cpu.peak_core_pct >= self.cpu_target_pct + 10.0:
            return max(self.min_workers, current_workers - 1)
        if cpu.overall_pct <= low_limit and cpu.peak_core_pct <= self.cpu_target_pct:
            return min(self.max_workers, current_workers + scale_step)
        if cpu.overall_pct <= self.cpu_target_pct - 8.0 and cpu.peak_core_pct <= self.cpu_target_pct:
            return min(self.max_workers, current_workers + 1)
        return current_workers


def _resolve_core_count() -> int:
    if psutil is not None:
        count = psutil.cpu_count(logical=True)
        if count:
            return int(count)
    return cpu_count() or 1


def _resolve_initial_workers(*, planned_max_workers: int, normalized_min_workers: int) -> int:
    return min(planned_max_workers, max(normalized_min_workers, min(4, planned_max_workers)))


def _sample_cpu_usage() -> CpuUsageSnapshot:
    if psutil is None:
        return CpuUsageSnapshot(
            overall_pct=0.0,
            average_core_pct=0.0,
            peak_core_pct=0.0,
            active_core_count=0,
            core_count=_resolve_core_count(),
        )

    per_core = [float(value) for value in psutil.cpu_percent(interval=None, percpu=True)]
    if not per_core:
        overall = float(psutil.cpu_percent(interval=None))
        core_count = _resolve_core_count()
        return CpuUsageSnapshot(
            overall_pct=overall,
            average_core_pct=overall,
            peak_core_pct=overall,
            active_core_count=1 if overall > 0 else 0,
            core_count=core_count,
        )

    overall_pct = float(sum(per_core) / len(per_core))
    active_cores = sum(1 for value in per_core if value >= 5.0)
    return CpuUsageSnapshot(
        overall_pct=overall_pct,
        average_core_pct=overall_pct,
        peak_core_pct=max(per_core),
        active_core_count=active_cores,
        core_count=len(per_core),
    )


def _update_progress_postfix(progress_bar, *, workers: int, pending: int, cpu: CpuUsageSnapshot) -> None:
    if hasattr(progress_bar, "set_postfix_str"):
        progress_bar.set_postfix_str(
            (
                f"workers={workers} "
                f"cpu={cpu.overall_pct:.0f}% "
                f"peak={cpu.peak_core_pct:.0f}% "
                f"active={cpu.active_core_count}/{max(1, cpu.core_count)} "
                f"pending={pending}"
            ),
            refresh=False,
        )
