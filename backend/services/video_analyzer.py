from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from backend.ai.association import associate_occupant
from backend.ai.detector import SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from backend.models.entities import AnalysisJob, Detection, Event, EventType, Evidence, InputScope, JobStatus, ReviewStatus, Video


class VideoAnalyzer:
    def __init__(self, detector: SafetyDetector, evidence_root: Path):
        self.detector = detector
        self.evidence_root = evidence_root

    def analyze(self, session: Session, job: AnalysisJob, video: Video) -> None:
        if video.input_scope != InputScope.VEHICLE_CABIN_CROP:
            raise ValueError("analysis requires a tracked vehicle/cabin crop")
        vehicle_context_id = f"video:{video.id}:cabin"
        job.status = JobStatus.RUNNING
        session.commit()
        capture = cv2.VideoCapture(video.storage_path)
        if not capture.isOpened():
            raise ValueError("video cannot be decoded")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        engine = TemporalEventEngine(**self.detector.config["temporal"])
        occupant_regions = self.detector.config["occupant_rois"]["regions"]
        frame_number = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_number / fps
                detections = self.detector.predict(frame, track=True)
                for detection in detections:
                    occupant_role = associate_occupant(
                        detection,
                        frame.shape[1],
                        frame.shape[0],
                        occupant_regions,
                    )
                    session.add(Detection(job_id=job.id, frame_number=frame_number, timestamp_seconds=timestamp, class_name=detection.class_name, confidence=detection.confidence, bbox={"xyxy": detection.xyxy}, track_id=detection.track_id, driver_associated=occupant_role == "driver", occupant_role=occupant_role, vehicle_context_id=vehicle_context_id))
                    candidates = engine.add(Observation(timestamp, detection.class_name, detection.confidence, detection.track_id, occupant_role, vehicle_context_id))
                    for candidate in candidates:
                        event = Event(job_id=job.id, video_id=video.id, event_type=EventType(candidate.event_type), confidence=candidate.confidence, frame_number=frame_number, timestamp_seconds=timestamp, track_id=candidate.track_id, occupant_role=candidate.occupant_role, vehicle_context_id=candidate.vehicle_context_id, model_version=self.detector.model_version, review_status=ReviewStatus(candidate.review_status))
                        session.add(event)
                        session.flush()
                        self._save_evidence(session, event, frame, detections)
                frame_number += 1
                if frame_number % 30 == 0:
                    job.progress = min(0.99, frame_number / total)
                    session.commit()
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.now(UTC)
            session.commit()
        finally:
            capture.release()

    def _save_evidence(self, session: Session, event: Event, frame, detections) -> None:
        root = self.evidence_root / event.id
        root.mkdir(parents=True, exist_ok=False)
        original = root / "original.jpg"
        annotated = root / "annotated.jpg"
        cv2.imwrite(str(original), frame)
        marked = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)
            cv2.rectangle(marked, (x1, y1), (x2, y2), (41, 123, 74), 2)
            cv2.putText(marked, f"{detection.class_name} {detection.confidence:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (41, 123, 74), 1, cv2.LINE_AA)
        cv2.imwrite(str(annotated), marked)
        sha = hashlib.sha256(original.read_bytes()).hexdigest()
        session.add(Evidence(event_id=event.id, original_path=str(original), annotated_path=str(annotated), sha256=sha))
