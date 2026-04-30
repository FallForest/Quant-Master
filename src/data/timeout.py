from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Thread
from time import monotonic
from typing import Callable, Generic, TypeVar


TResult = TypeVar("TResult")


@dataclass(frozen=True)
class TimeoutResult(Generic[TResult]):
    value: TResult
    elapsed_seconds: float


class SubtaskTimeoutError(TimeoutError):
    def __init__(self, *, task_label: str, timeout_seconds: float) -> None:
        self.task_label = str(task_label)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(f"{self.task_label} timed out after {self.timeout_seconds:.1f}s")


def run_with_timeout(
    fn: Callable[[], TResult],
    *,
    timeout_seconds: float | None,
    task_label: str,
) -> TimeoutResult[TResult]:
    if timeout_seconds is None:
        started_at = monotonic()
        return TimeoutResult(value=fn(), elapsed_seconds=monotonic() - started_at)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0 when provided.")

    queue: Queue[tuple[bool, TResult | None, Exception | None]] = Queue(maxsize=1)

    def _runner() -> None:
        try:
            queue.put((True, fn(), None))
        except Exception as exc:  # noqa: BLE001
            queue.put((False, None, exc))

    thread = Thread(target=_runner, name=f"timeout:{task_label}", daemon=True)
    started_at = monotonic()
    thread.start()
    thread.join(timeout_seconds)
    elapsed_seconds = monotonic() - started_at
    if thread.is_alive():
        raise SubtaskTimeoutError(task_label=task_label, timeout_seconds=timeout_seconds)
    succeeded, value, error = queue.get()
    if not succeeded:
        assert error is not None
        raise error
    return TimeoutResult(value=value, elapsed_seconds=elapsed_seconds)
