from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar


TItem = TypeVar("TItem")
TResult = TypeVar("TResult")
TContext = TypeVar("TContext")


@dataclass(frozen=True)
class ThreadTaskOutcome(Generic[TItem, TResult, TContext]):
    index: int
    item: TItem
    context: TContext | None
    value: TResult | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ThreadTaskReport(Generic[TItem, TResult, TContext]):
    outcomes: list[ThreadTaskOutcome[TItem, TResult, TContext]]

    @property
    def failures(self) -> list[ThreadTaskOutcome[TItem, TResult, TContext]]:
        return [outcome for outcome in self.outcomes if not outcome.succeeded]


@dataclass(frozen=True)
class _RunningThreadTask(Generic[TItem, TContext]):
    index: int
    item: TItem
    context: TContext | None


def run_bounded_thread_pool(
    *,
    items: Sequence[TItem],
    max_workers: int,
    submitter: Callable[[TItem], tuple[Callable[[], TResult], TContext | None] | None],
    on_complete: Callable[[ThreadTaskOutcome[TItem, TResult, TContext]], None] | None = None,
    continue_on_error: bool = False,
) -> ThreadTaskReport[TItem, TResult, TContext]:
    item_list = list(items)
    if not item_list:
        return ThreadTaskReport(outcomes=[])
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")

    effective_workers = min(int(max_workers), len(item_list))
    pending: deque[tuple[int, TItem]] = deque(enumerate(item_list))
    outcomes: list[ThreadTaskOutcome[TItem, TResult, TContext] | None] = [None] * len(item_list)
    running: dict[Future, _RunningThreadTask[TItem, TContext]] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        while pending or running:
            blocked = 0
            while pending and len(running) < effective_workers and blocked < len(pending):
                index, item = pending.popleft()
                submission = submitter(item)
                if submission is None:
                    pending.append((index, item))
                    blocked += 1
                    continue

                task_fn, context = submission
                future = executor.submit(task_fn)
                running[future] = _RunningThreadTask(index=index, item=item, context=context)
                blocked = 0

            if not running:
                if pending:
                    raise RuntimeError("No pending task could be submitted. Check resource gating logic.")
                break

            completed, _ = wait(tuple(running.keys()), return_when=FIRST_COMPLETED)
            for future in completed:
                task = running.pop(future)
                try:
                    outcome = ThreadTaskOutcome(
                        index=task.index,
                        item=task.item,
                        context=task.context,
                        value=future.result(),
                    )
                except Exception as exc:  # noqa: BLE001
                    outcome = ThreadTaskOutcome(
                        index=task.index,
                        item=task.item,
                        context=task.context,
                        error=exc,
                    )
                outcomes[task.index] = outcome
                if on_complete is not None:
                    on_complete(outcome)
                if outcome.error is not None and not continue_on_error:
                    raise outcome.error

    return ThreadTaskReport(outcomes=[outcome for outcome in outcomes if outcome is not None])
