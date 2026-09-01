from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from backend.ai.detector import SafetyDetector
from backend.models.entities import AnalysisJob, JobStatus, Video
from backend.services.video_analyzer import VideoAnalyzer


class InferenceWorker:
    """Own analysis execution independently of HTTP scheduling."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        detector: SafetyDetector,
        evidence_root: Path,
    ):
        self.session_factory = session_factory
        self.detector = detector
        self.evidence_root = evidence_root

    def run(self, job_id: str) -> None:
        with self.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if not job:
                return
            video = session.get(Video, job.video_id)
            if not video:
                job.status = JobStatus.FAILED
                job.error = "video record is missing"
                job.completed_at = datetime.now(UTC)
                session.commit()
                return
            try:
                VideoAnalyzer(self.detector, self.evidence_root).analyze(session, job, video)
            except Exception as exc:
                session.rollback()
                job = session.get(AnalysisJob, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error = str(exc)[:2000]
                    job.completed_at = datetime.now(UTC)
                    session.commit()
