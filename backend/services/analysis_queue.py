from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from fastapi import BackgroundTasks
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.models.entities import AnalysisJob, JobStatus


class AnalysisQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class BackgroundTaskAnalysisQueue:
    """Demo adapter. Replace with a durable queue without changing analysis business logic."""

    def __init__(self, tasks: BackgroundTasks, worker: Callable[[str], None]):
        self.tasks = tasks
        self.worker = worker

    def enqueue(self, job_id: str) -> None:
        self.tasks.add_task(self.worker, job_id)


class DurableAnalysisQueue:
    """Database-backed queue whose queued jobs survive API process restarts."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def enqueue(self, job_id: str) -> None:
        with self.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if not job or job.status != JobStatus.QUEUED:
                raise ValueError("durable queue requires a persisted QUEUED job")

    def claim_next(self) -> str | None:
        """Atomically claim the oldest queued job; competing workers receive another job."""
        with self.session_factory() as session:
            job_id = session.scalar(
                select(AnalysisJob.id)
                .where(AnalysisJob.status == JobStatus.QUEUED)
                .order_by(AnalysisJob.created_at, AnalysisJob.id)
                .limit(1)
            )
            if job_id is None:
                return None
            result = session.execute(
                update(AnalysisJob)
                .where(AnalysisJob.id == job_id, AnalysisJob.status == JobStatus.QUEUED)
                .values(status=JobStatus.RUNNING, error=None)
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            session.commit()
            return job_id
