from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from backend.ai.association import associate_occupant
from backend.ai.auxiliary import (
    PoseEstimator,
    SeatbeltClassifier,
    VehicleDetector,
    crop_box,
    phone_context,
)
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from backend.ai.fusion import FusionFeatures, LogisticFusion
from backend.ai.sequence import TemporalFeatureBuffer
from backend.models.entities import (
    AnalysisJob,
    Detection,
    Event,
    EventType,
    Evidence,
    InputScope,
    JobStatus,
    ReviewStatus,
    Video,
)


class VideoAnalyzer:
    def __init__(self, detector: SafetyDetector, evidence_root: Path):
        self.detector = detector
        self.evidence_root = evidence_root
        auxiliary = detector.config.get("auxiliary_models", {})
        pose_path = Path(auxiliary["pose_weights"]) if auxiliary.get("pose_weights") else None
        seatbelt_path = (
            Path(auxiliary["seatbelt_classifier_weights"])
            if auxiliary.get("seatbelt_classifier_weights")
            else None
        )
        fusion_path = Path(auxiliary["fusion_artifact"]) if auxiliary.get("fusion_artifact") else None
        self.pose = PoseEstimator(pose_path, float(auxiliary.get("pose_confidence", 0.25)))
        self.seatbelt_classifier = SeatbeltClassifier(seatbelt_path)
        self.fusion = LogisticFusion(fusion_path, float(auxiliary.get("fusion_threshold", 0.5)))
        self.use_fusion = bool(auxiliary.get("enabled", False))
        vehicle = detector.config.get("vehicle_context", {})
        vehicle_path = Path(vehicle["vehicle_weights"]) if vehicle.get("vehicle_weights") else None
        self.vehicle_detector = VehicleDetector(
            vehicle_path, float(vehicle.get("vehicle_confidence", 0.35))
        )
        temporal = detector.config.get("temporal", {})
        self.sequence = TemporalFeatureBuffer(
            window_seconds=float(temporal.get("window_seconds", 2.5)),
            alpha=float(temporal.get("smoothing_alpha", 0.35)),
            positive_score=float(temporal.get("feature_positive_score", 0.5)),
        )
        seatbelt_policy = detector.config.get("seatbelt_policy", {})
        self.seatbelt_min_confidence = float(
            seatbelt_policy.get("classifier_min_confidence", 0.60)
        )
        self.seatbelt_min_margin = float(seatbelt_policy.get("classifier_min_margin", 0.15))

    def supports_scope(self, scope: InputScope) -> bool:
        if scope == InputScope.VEHICLE_CABIN_CROP:
            return True
        return scope == InputScope.TRAFFIC_SCENE_WITH_VEHICLE_ROIS and self.vehicle_detector.available

    def _frame_contexts(self, frame, video: Video):
        if video.input_scope == InputScope.VEHICLE_CABIN_CROP:
            yield f"video:{video.id}:cabin", frame, (0, 0), None
            return
        for region in self.vehicle_detector.predict(frame):
            x1, y1, x2, y2 = map(int, region.xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            context_id = f"video:{video.id}:{region.context_id}"
            yield context_id, frame[y1:y2, x1:x2], (x1, y1), region.track_id

    def analyze(self, session: Session, job: AnalysisJob, video: Video) -> None:
        if not self.supports_scope(video.input_scope):
            raise ValueError("input scope requires configured local vehicle detector weights")
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
                for vehicle_context_id, cabin, offset, vehicle_track_id in self._frame_contexts(
                    frame, video
                ):
                    local_detections = self.detector.predict(
                        cabin, track=video.input_scope == InputScope.VEHICLE_CABIN_CROP
                    )
                    poses = self.pose.predict(cabin) if self.use_fusion else []
                    for detection in local_detections:
                        seatbelt_evidence = None
                        roi_evidence_source = "DETECTOR_ONLY"
                        if detection.class_name == "occupant_upper_body":
                            seatbelt_evidence = self.seatbelt_classifier.predict(
                                crop_box(cabin, detection.xyxy)
                            )
                            if seatbelt_evidence is None:
                                detection = replace(
                                    detection, class_id=-1, class_name="seatbelt_uncertain"
                                )
                                roi_evidence_source = "CLASSIFIER_UNAVAILABLE"
                            else:
                                states = {
                                    "seatbelt_fastened": seatbelt_evidence.fastened,
                                    "seatbelt_unfastened": seatbelt_evidence.unfastened,
                                    "seatbelt_uncertain": seatbelt_evidence.uncertain_or_occluded,
                                }
                                ranked_states = sorted(
                                    states.items(), key=lambda item: item[1], reverse=True
                                )
                                state, state_score = ranked_states[0]
                                margin = state_score - ranked_states[1][1]
                                if (
                                    state_score < self.seatbelt_min_confidence
                                    or margin < self.seatbelt_min_margin
                                ):
                                    state = "seatbelt_uncertain"
                                class_id = {
                                    "seatbelt_fastened": 1,
                                    "seatbelt_unfastened": 2,
                                    "seatbelt_uncertain": -1,
                                }[state]
                                detection = replace(
                                    detection,
                                    class_id=class_id,
                                    class_name=state,
                                    confidence=detection.confidence * state_score,
                                )
                                roi_evidence_source = (
                                    "CLASSIFIER_UNCERTAIN"
                                    if state == "seatbelt_uncertain"
                                    else "ROI_DETECTOR_CLASSIFIER"
                                )
                        occupant_role = associate_occupant(
                            detection,
                            cabin.shape[1],
                            cabin.shape[0],
                            occupant_regions,
                        )
                        context = "UNKNOWN"
                        hand_proximity = face_proximity = pose_confidence = 0.0
                        if detection.class_name == "phone" and poses:
                            matching_poses = []
                            for pose in poses:
                                if pose.xyxy is None:
                                    continue
                                pose_detection = NormalizedDetection(
                                    -1, "person_pose", pose.confidence, pose.xyxy
                                )
                                pose_role = associate_occupant(
                                    pose_detection,
                                    cabin.shape[1],
                                    cabin.shape[0],
                                    occupant_regions,
                                )
                                if pose_role == occupant_role:
                                    matching_poses.append(pose)
                            pose_options = [
                                phone_context(detection, pose) for pose in matching_poses
                            ]
                            if not pose_options:
                                pose_options = [("UNKNOWN", 0.0, 0.0)]
                            context, hand_proximity, face_proximity = max(
                                pose_options, key=lambda item: max(item[1], item[2])
                            )
                            pose_confidence = max(
                                (pose.confidence for pose in matching_poses), default=0.0
                            )
                        classifier_fastened = classifier_unfastened = 0.0
                        evidence_source = roi_evidence_source
                        if detection.class_name.startswith("seatbelt_") and self.use_fusion:
                            evidence = seatbelt_evidence or self.seatbelt_classifier.predict(
                                crop_box(cabin, detection.xyxy)
                            )
                            if evidence:
                                classifier_fastened = evidence.fastened
                                classifier_unfastened = evidence.unfastened
                                evidence_source = evidence.source
                        fusion_score = None
                        sequence_key = (
                            vehicle_context_id,
                            detection.track_id,
                            occupant_role,
                            detection.class_name,
                        )
                        sequence = self.sequence.add(
                            sequence_key, timestamp, detection.confidence
                        )
                        if self.use_fusion and detection.class_name in {
                            "phone",
                            "seatbelt_fastened",
                            "seatbelt_unfastened",
                        }:
                            result = self.fusion.predict(
                                FusionFeatures(
                                    detector_confidence=detection.confidence,
                                    classifier_unfastened=classifier_unfastened,
                                    classifier_fastened=classifier_fastened,
                                    phone_hand_proximity=hand_proximity,
                                    phone_face_proximity=face_proximity,
                                    pose_confidence=pose_confidence,
                                    temporal_ratio=sequence.ratio,
                                    driver_role=1.0 if occupant_role == "driver" else 0.0,
                                    vehicle_context=1.0,
                                )
                            )
                            fusion_score = result.score
                            if not result.decision:
                                evidence_source = "FUSION_REJECTED"
                            elif evidence_source == "DETECTOR_ONLY":
                                evidence_source = result.source
                        track_id = detection.track_id
                        if track_id is None and vehicle_track_id is not None:
                            track_id = vehicle_track_id * 10 + detection.class_id
                        x_offset, y_offset = offset
                        global_detection = replace(
                            detection,
                            xyxy=(
                                detection.xyxy[0] + x_offset,
                                detection.xyxy[1] + y_offset,
                                detection.xyxy[2] + x_offset,
                                detection.xyxy[3] + y_offset,
                            ),
                            track_id=track_id,
                        )
                        session.add(
                            Detection(
                                job_id=job.id,
                                frame_number=frame_number,
                                timestamp_seconds=timestamp,
                                class_name=detection.class_name,
                                confidence=detection.confidence,
                                bbox={
                                    "xyxy": global_detection.xyxy,
                                    "phone_context": context,
                                    "fusion_score": fusion_score,
                                    "evidence_source": evidence_source,
                                },
                                track_id=track_id,
                                driver_associated=occupant_role == "driver",
                                occupant_role=occupant_role,
                                vehicle_context_id=vehicle_context_id,
                            )
                        )
                        candidates = engine.add(
                            Observation(
                                timestamp,
                                detection.class_name,
                                detection.confidence,
                                track_id,
                                occupant_role,
                                vehicle_context_id,
                                context,
                                fusion_score,
                                evidence_source,
                            )
                        )
                        for candidate in candidates:
                            event = Event(
                                job_id=job.id,
                                video_id=video.id,
                                event_type=EventType(candidate.event_type),
                                confidence=candidate.confidence,
                                frame_number=frame_number,
                                timestamp_seconds=timestamp,
                                track_id=candidate.track_id,
                                occupant_role=candidate.occupant_role,
                                vehicle_context_id=candidate.vehicle_context_id,
                                model_version=self.detector.model_version,
                                review_status=ReviewStatus(candidate.review_status),
                            )
                            session.add(event)
                            session.flush()
                            self._save_evidence(session, event, frame, [global_detection])
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
