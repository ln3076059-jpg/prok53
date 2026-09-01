from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from fastapi import BackgroundTasks


class AnalysisQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class BackgroundTaskAnalysisQueue:
    """Demo adapter. Replace with a durable queue without changing analysis business logic."""

    def __init__(self, tasks: BackgroundTasks, worker: Callable[[str], None]):
        self.tasks = tasks
        self.worker = worker

    def enqueue(self, job_id: str) -> None:
        self.tasks.add_task(self.worker, job_id)
