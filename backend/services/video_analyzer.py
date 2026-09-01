from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from backend.ai.association import OccupantAssignment, OccupantAssociator
from backend.ai.auxiliary import (
    PoseEstimator,
    SeatbeltClassifier,
    SeatbeltEvidence,
    VehicleDetector,
    classify_phone_context,
    crop_box,
)
from backend.ai.cabin import CabinLocalizer, crop_cabin
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from backend.ai.evidence import EventEvidenceBuffer, EvidenceArtifacts, annotate_frame
from backend.ai.fusion import FusionFeatures, LogisticFusion
from backend.ai.sequence import TemporalFeatureBuffer
from backend.ai.tracking import WithinVehicleTracker
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
from backend.services.video_source import FileVideoSource


@dataclass(frozen=True)
class InferenceContext:
    context_id: str
    cabin: object
    offset: tuple[int, int]
    vehicle_track_id: int | None
    vehicle_type: str
    vehicle_confidence: float
    cabin_confidence: float
    cabin_method: str


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
        fusion_path = (
            Path(auxiliary["fusion_artifact"]) if auxiliary.get("fusion_artifact") else None
        )
        self.pose = PoseEstimator(pose_path, float(auxiliary.get("pose_confidence", 0.25)))
        self.seatbelt_classifier = SeatbeltClassifier(seatbelt_path)
        activation = detector.config.get("activation_policy", {})
        self.fusion = LogisticFusion(
            fusion_path,
            float(auxiliary.get("fusion_threshold", 0.5)),
            require_calibrated=bool(activation.get("require_calibrated_fusion", False)),
        )
        self.use_fusion = bool(auxiliary.get("enabled", False))

        vehicle = detector.config.get("vehicle_context", {})
        vehicle_path = Path(vehicle["vehicle_weights"]) if vehicle.get("vehicle_weights") else None
        self.vehicle_detector = VehicleDetector(
            vehicle_path, float(vehicle.get("vehicle_confidence", 0.35))
        )
        self.cabin_localizer = CabinLocalizer(vehicle.get("cabin_localizer", {}))
        self.camera_cabin_calibration = vehicle.get("camera_cabin_calibration")

        occupant_config = detector.config.get("occupant_association", {})
        roi_config = detector.config.get("occupant_rois", {})
        calibrated_regions = (
            roi_config.get("regions", [])
            if str(roi_config.get("calibration_status", "")).startswith("APPROVED")
            else []
        )
        self.occupants = OccupantAssociator(
            driver_side=str(occupant_config.get("image_driver_side", "left")),
            minimum_confidence=float(occupant_config.get("minimum_confidence", 0.55)),
            rear_seats_enabled=bool(occupant_config.get("rear_seats_enabled", True)),
            calibrated_regions=calibrated_regions,
        )
        tracking = detector.config.get("tracking", {})
        self.local_tracker = WithinVehicleTracker(
            max_gap_frames=int(tracking.get("max_gap_frames", 12)),
            minimum_iou=float(tracking.get("minimum_iou", 0.10)),
            maximum_center_distance=float(tracking.get("maximum_center_distance", 0.25)),
        )

        temporal = detector.config.get("temporal", {})
        self.sequence = TemporalFeatureBuffer(
            window_seconds=float(temporal.get("window_seconds", 2.5)),
            alpha=float(temporal.get("smoothing_alpha", 0.35)),
            positive_score=float(temporal.get("feature_positive_score", 0.5)),
        )
        seatbelt_policy = detector.config.get("seatbelt_policy", {})
        self.seatbelt_min_confidence = float(seatbelt_policy.get("classifier_min_confidence", 0.60))
        self.seatbelt_min_margin = float(seatbelt_policy.get("classifier_min_margin", 0.15))
        performance = detector.config.get("performance", {})
        self.behavior_interval = max(1, int(performance.get("behavior_detector_every_n_frames", 1)))
        self.vehicle_interval = max(1, int(performance.get("vehicle_detector_every_n_frames", 1)))
        self.pose_interval = max(1, int(performance.get("pose_every_n_frames", 2)))
        self.classifier_interval = max(
            1, int(performance.get("seatbelt_classifier_every_n_frames", 1))
        )
        self._vehicle_region_cache = []
        self._seatbelt_cache: dict[tuple[str, int | None], SeatbeltEvidence | None] = {}
        self._feature_history: defaultdict[tuple, deque[dict]] = defaultdict(
            lambda: deque(maxlen=120)
        )
        evidence = detector.config.get("evidence", {})
        self.evidence_pre_seconds = float(evidence.get("pre_seconds", 2.0))
        self.evidence_post_seconds = float(evidence.get("post_seconds", 2.5))

    def runtime_status(self) -> dict:
        fusion_status = self.fusion.status()
        fusion_status["fusion_enabled"] = self.use_fusion
        if not self.use_fusion:
            fusion_status["fusion_mode"] = "DISABLED"
            fusion_status["fusion_fail_closed"] = False
        return {
            "fusion": fusion_status,
            "cabin_localizer": {
                "custom_detector_available": self.cabin_localizer.detector_available,
                "minimum_confidence": self.cabin_localizer.minimum_confidence,
                "fallback_policy": "UNKNOWN_CABIN_FAIL_CLOSED",
            },
            "inference_schedule": {
                "vehicle_detector_every_n_frames": self.vehicle_interval,
                "behavior_detector_every_n_frames": self.behavior_interval,
                "pose_every_n_frames": self.pose_interval,
                "seatbelt_classifier_every_n_frames": self.classifier_interval,
            },
            "streaming": "EXPERIMENTAL_NOT_ACTIVE_FOR_UPLOAD_JOBS",
        }

    def supports_scope(self, scope: InputScope) -> bool:
        if scope == InputScope.VEHICLE_CABIN_CROP:
            return True
        return (
            scope == InputScope.TRAFFIC_SCENE_WITH_VEHICLE_ROIS and self.vehicle_detector.available
        )

    def _frame_contexts(self, frame, video: Video, frame_number: int = 0) -> list[InferenceContext]:
        if video.input_scope == InputScope.VEHICLE_CABIN_CROP:
            return [
                InferenceContext(
                    f"video:{video.id}:provided-cabin",
                    frame,
                    (0, 0),
                    None,
                    "unknown",
                    1.0,
                    1.0,
                    "PROVIDED_CABIN_INPUT_SCOPE",
                )
            ]

        if frame_number % self.vehicle_interval == 0 or not self._vehicle_region_cache:
            self._vehicle_region_cache = self.vehicle_detector.predict(frame)

        contexts: list[InferenceContext] = []
        for region in self._vehicle_region_cache:
            # Raw traffic inputs need a stable outer-vehicle identity. Never build
            # temporal evidence on a list index or an untracked detector result.
            if region.track_id is None:
                continue
            x1, y1, x2, y2 = map(int, region.xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            vehicle_crop = frame[y1:y2, x1:x2]
            cabin_region = self.cabin_localizer.locate(
                vehicle_crop, region.vehicle_class, self.camera_cabin_calibration
            )
            if not cabin_region.known:
                # Unknown cabin is deliberately not rebranded as a valid vehicle context.
                continue
            cabin, cabin_offset = crop_cabin(vehicle_crop, cabin_region)
            if cabin.size == 0:
                continue
            contexts.append(
                InferenceContext(
                    f"video:{video.id}:{region.context_id}",
                    cabin,
                    (x1 + cabin_offset[0], y1 + cabin_offset[1]),
                    region.track_id,
                    region.vehicle_class,
                    region.confidence,
                    cabin_region.confidence,
                    cabin_region.method,
                )
            )
        return contexts

    def analyze(self, session: Session, job: AnalysisJob, video: Video) -> None:
        if not self.supports_scope(video.input_scope):
            raise ValueError("input scope requires configured local vehicle detector weights")
        job.status = JobStatus.RUNNING
        session.commit()
        source = FileVideoSource(video.storage_path)
        total, fps = source.frame_count, source.fps
        evidence_buffer = EventEvidenceBuffer(
            fps, self.evidence_pre_seconds, self.evidence_post_seconds
        )
        engine = TemporalEventEngine(**self.detector.config.get("temporal", {}))
        frame_number = 0
        previous_contexts: set[str] = set()
        pose_cache: dict[str, list] = {}
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                timestamp = frame_number / fps
                self._persist_artifacts(session, evidence_buffer.add_frame(timestamp, frame))
                contexts = self._frame_contexts(frame, video, frame_number)
                active_contexts = {context.context_id for context in contexts}
                for lost_context in previous_contexts - active_contexts:
                    engine.reset_vehicle(lost_context)
                    self.local_tracker.reset_vehicle(lost_context)
                    pose_cache.pop(lost_context, None)
                    self._reset_context_caches(lost_context)
                previous_contexts = active_contexts

                if frame_number % self.behavior_interval == 0:
                    for context in contexts:
                        self._analyze_context(
                            session,
                            job,
                            video,
                            frame,
                            frame_number,
                            timestamp,
                            context,
                            engine,
                            evidence_buffer,
                            pose_cache,
                        )
                engine.expire(timestamp)
                frame_number += 1
                if frame_number % 30 == 0:
                    job.progress = min(0.99, frame_number / total)
                    session.commit()

            self._persist_artifacts(session, evidence_buffer.flush())
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.now(UTC)
            session.commit()
        finally:
            source.close()

    def _analyze_context(
        self,
        session: Session,
        job: AnalysisJob,
        video: Video,
        full_frame,
        frame_number: int,
        timestamp: float,
        context: InferenceContext,
        engine: TemporalEventEngine,
        evidence_buffer: EventEvidenceBuffer,
        pose_cache: dict[str, list],
    ) -> None:
        cabin = context.cabin
        detections = self.detector.predict(cabin, track=False)
        detections = self.local_tracker.update(
            context.context_id,
            frame_number,
            detections,
            cabin.shape[1],
            cabin.shape[0],
        )
        if frame_number % self.pose_interval == 0:
            pose_cache[context.context_id] = self.pose.predict(cabin) if self.use_fusion else []
        poses = pose_cache.get(context.context_id, [])

        upper_bodies = [item for item in detections if item.class_name == "occupant_upper_body"]
        assigned_upper_bodies: list[tuple[NormalizedDetection, OccupantAssignment]] = [
            (
                item,
                self.occupants.assign_upper_body(item, cabin.shape[1], cabin.shape[0]),
            )
            for item in upper_bodies
        ]

        for raw_detection in detections:
            detection = raw_detection
            if detection.class_name == "occupant_upper_body":
                assignment = next(
                    assignment
                    for occupant, assignment in assigned_upper_bodies
                    if occupant.track_id == detection.track_id
                )
            else:
                assignment = self.occupants.assign_object(
                    detection,
                    assigned_upper_bodies,
                    cabin.shape[1],
                    cabin.shape[0],
                )

            seatbelt_evidence = None
            evidence_source = "DETECTOR_ONLY"
            detector_class_before_classifier = detection.class_name
            if detection.class_name == "occupant_upper_body":
                seatbelt_evidence = self._seatbelt_evidence(
                    context.context_id, frame_number, detection, cabin
                )
                detection, evidence_source = self._classify_seatbelt(detection, seatbelt_evidence)
            elif detection.class_name in {"seatbelt_fastened", "seatbelt_unfastened"}:
                seatbelt_evidence = self._seatbelt_evidence(
                    context.context_id, frame_number, detection, cabin
                )
                if seatbelt_evidence is not None:
                    predicted = (
                        "seatbelt_unfastened"
                        if seatbelt_evidence.unfastened > seatbelt_evidence.fastened
                        else "seatbelt_fastened"
                    )
                    confidence = max(seatbelt_evidence.unfastened, seatbelt_evidence.fastened)
                    if (
                        predicted != detection.class_name
                        and confidence >= self.seatbelt_min_confidence
                    ):
                        evidence_source = "DETECTOR_CLASSIFIER_CONFLICT"

            phone_state = "UNKNOWN_PHONE_CONTEXT"
            hand_proximity = face_proximity = pose_confidence = 0.0
            pose_points = {"face": [], "hands": []}
            if detection.class_name == "phone":
                pose_options = []
                for pose in poses:
                    if pose.xyxy is None:
                        continue
                    pose_detection = NormalizedDetection(
                        -1, "person_pose", pose.confidence, pose.xyxy
                    )
                    pose_assignment = self.occupants.assign_upper_body(
                        pose_detection, cabin.shape[1], cabin.shape[0]
                    )
                    if assignment.role == "unknown" or pose_assignment.role == assignment.role:
                        pose_options.append((pose, classify_phone_context(detection, pose)))
                if pose_options:
                    pose, result = max(pose_options, key=lambda item: max(item[1][1], item[1][2]))
                    phone_state, hand_proximity, face_proximity = result
                    pose_confidence = pose.confidence
                    pose_points = {
                        "face": [list(point) for point in pose.face_points],
                        "hands": [list(point) for point in pose.hand_points],
                    }

            classifier_fastened = seatbelt_evidence.fastened if seatbelt_evidence else 0.0
            classifier_unfastened = seatbelt_evidence.unfastened if seatbelt_evidence else 0.0
            sequence_key = (
                context.context_id,
                detection.track_id,
                assignment.role,
                detection.class_name,
            )
            sequence = self.sequence.add(sequence_key, timestamp, detection.confidence)
            fusion_score = None
            fusion_source = "NOT_APPLIED"
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
                        driver_role=1.0 if assignment.role == "driver" else 0.0,
                        vehicle_context=context.cabin_confidence * context.vehicle_confidence,
                    )
                )
                fusion_score, fusion_source = result.score, result.source
                if not result.decision:
                    evidence_source = "FUSION_REJECTED"
                elif evidence_source == "DETECTOR_ONLY":
                    evidence_source = result.source

            self._feature_history[sequence_key].append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_number": frame_number,
                    "detector_confidence": detection.confidence,
                    "fusion_score": fusion_score,
                    "fusion_source": fusion_source,
                    "phone_context": phone_state,
                    "phone_hand_proximity": hand_proximity,
                    "phone_face_proximity": face_proximity,
                    "pose_confidence": pose_confidence,
                    "pose_points": pose_points,
                    "seatbelt_probabilities": (
                        {
                            "fastened": classifier_fastened,
                            "unfastened": classifier_unfastened,
                            "uncertain_or_occluded": seatbelt_evidence.uncertain_or_occluded,
                        }
                        if seatbelt_evidence
                        else None
                    ),
                    "occupant_role": assignment.role,
                    "occupant_role_confidence": assignment.confidence,
                    "vehicle_context_confidence": context.cabin_confidence
                    * context.vehicle_confidence,
                    "temporal_positive_ratio": sequence.positive_ratio,
                    "temporal_duration_seconds": sequence.duration_seconds,
                    "temporal_maximum_gap_seconds": sequence.maximum_gap_seconds,
                }
            )

            global_detection = self._global_detection(detection, context.offset)
            probabilities = (
                (
                    seatbelt_evidence.fastened,
                    seatbelt_evidence.unfastened,
                    seatbelt_evidence.uncertain_or_occluded,
                )
                if seatbelt_evidence
                else None
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
                        "phone_context": phone_state,
                        "phone_hand_proximity": hand_proximity,
                        "phone_face_proximity": face_proximity,
                        "pose_confidence": pose_confidence,
                        "seatbelt_probabilities": probabilities,
                        "fusion_score": fusion_score,
                        "fusion_source": fusion_source,
                        "evidence_source": evidence_source,
                        "vehicle_type": context.vehicle_type,
                        "vehicle_confidence": context.vehicle_confidence,
                        "cabin_confidence": context.cabin_confidence,
                        "cabin_method": context.cabin_method,
                        "occupant_role_confidence": assignment.confidence,
                        "occupant_association_method": assignment.method,
                        "detector_class_before_classifier": detector_class_before_classifier,
                    },
                    track_id=detection.track_id,
                    driver_associated=assignment.role == "driver",
                    occupant_role=assignment.role,
                    vehicle_context_id=context.context_id,
                )
            )
            candidates = engine.add(
                Observation(
                    timestamp=timestamp,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    track_id=detection.track_id,
                    occupant_role=assignment.role,
                    vehicle_context_id=context.context_id,
                    phone_context=phone_state,
                    fusion_score=fusion_score,
                    evidence_source=evidence_source,
                    vehicle_type=context.vehicle_type,
                    occupant_role_confidence=assignment.confidence,
                    vehicle_context_confidence=context.cabin_confidence
                    * context.vehicle_confidence,
                    phone_hand_proximity=hand_proximity,
                    phone_face_proximity=face_proximity,
                    pose_confidence=pose_confidence,
                    seatbelt_probabilities=probabilities,
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
                evidence_buffer.register(
                    event.id,
                    self.evidence_root,
                    timestamp,
                    full_frame,
                    annotate_frame(full_frame, [global_detection]),
                    {
                        "event_id": event.id,
                        "event_type": candidate.event_type,
                        "timestamp_seconds": timestamp,
                        "frame_number": frame_number,
                        "vehicle_id": context.context_id,
                        "vehicle_type": context.vehicle_type,
                        "vehicle_confidence": context.vehicle_confidence,
                        "cabin_confidence": context.cabin_confidence,
                        "cabin_method": context.cabin_method,
                        "occupant_role": candidate.occupant_role,
                        "occupant_role_confidence": assignment.confidence,
                        "occupant_association_method": assignment.method,
                        "track_id": candidate.track_id,
                        "phone_context": phone_state,
                        "phone_hand_proximity": hand_proximity,
                        "phone_face_proximity": face_proximity,
                        "pose_confidence": pose_confidence,
                        "seatbelt_probabilities": probabilities,
                        "fusion_score": fusion_score,
                        "fusion_mode": self.fusion.status()["fusion_mode"],
                        "temporal_score": candidate.temporal_score,
                        "model_version": self.detector.model_version,
                        "model_config_sha256": self.detector.config_sha256,
                        "thresholds": self.detector.thresholds,
                        "review_status_at_creation": candidate.review_status,
                        "temporal_observations": list(self._feature_history[sequence_key]),
                    },
                )

    def _seatbelt_evidence(
        self,
        context_id: str,
        frame_number: int,
        detection: NormalizedDetection,
        cabin,
    ) -> SeatbeltEvidence | None:
        key = (context_id, detection.track_id)
        if frame_number % self.classifier_interval == 0 or key not in self._seatbelt_cache:
            self._seatbelt_cache[key] = self.seatbelt_classifier.predict(
                crop_box(cabin, detection.xyxy)
            )
        return self._seatbelt_cache[key]

    def _reset_context_caches(self, context_id: str) -> None:
        self.sequence.reset_prefix((context_id,))
        for key in [key for key in self._seatbelt_cache if key[0] == context_id]:
            self._seatbelt_cache.pop(key, None)
        for key in [key for key in self._feature_history if key[0] == context_id]:
            self._feature_history.pop(key, None)

    def _classify_seatbelt(self, detection, evidence):
        if evidence is None:
            return (
                replace(detection, class_id=-1, class_name="seatbelt_uncertain"),
                "CLASSIFIER_UNAVAILABLE",
            )
        states = {
            "seatbelt_fastened": evidence.fastened,
            "seatbelt_unfastened": evidence.unfastened,
            "seatbelt_uncertain": evidence.uncertain_or_occluded,
        }
        ranked = sorted(states.items(), key=lambda item: item[1], reverse=True)
        state, score = ranked[0]
        margin = score - ranked[1][1]
        source = "ROI_DETECTOR_CLASSIFIER"
        if score < self.seatbelt_min_confidence or margin < self.seatbelt_min_margin:
            state, source = "seatbelt_uncertain", "CLASSIFIER_UNCERTAIN"
        class_id = {"seatbelt_fastened": 1, "seatbelt_unfastened": 2, "seatbelt_uncertain": -1}[
            state
        ]
        return replace(
            detection, class_id=class_id, class_name=state, confidence=detection.confidence * score
        ), source

    @staticmethod
    def _global_detection(detection: NormalizedDetection, offset: tuple[int, int]):
        x_offset, y_offset = offset
        return replace(
            detection,
            xyxy=(
                detection.xyxy[0] + x_offset,
                detection.xyxy[1] + y_offset,
                detection.xyxy[2] + x_offset,
                detection.xyxy[3] + y_offset,
            ),
        )

    @staticmethod
    def _persist_artifacts(session: Session, artifacts: list[EvidenceArtifacts]) -> None:
        for item in artifacts:
            session.add(
                Evidence(
                    event_id=item.event_id,
                    original_path=str(item.original_path),
                    annotated_path=str(item.annotated_path),
                    sha256=item.hashes["original.jpg"],
                )
            )
