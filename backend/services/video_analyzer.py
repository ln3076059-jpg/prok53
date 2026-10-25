from __future__ import annotations

import gc
import json
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from sqlalchemy.orm import Session

try:
    torch.set_num_threads(2)
except Exception:
    pass

from backend.ai.association import (
    OccupantAssignment,
    OccupantAssociator,
    intersection_over_box,
)
from backend.ai.auxiliary import (
    PoseEstimator,
    SeatbeltClassifier,
    SeatbeltEvidence,
    VehicleDetector,
    calibrate_v32_phone_confidence,
    calibrate_v32_seatbelt_confidence,
    crop_box,
    evaluate_occupant_violation_policy,
    infer_phone_posture_and_action,
    is_valid_phone_aspect_ratio,
    reason_phone_context,
)
from backend.ai.cabin import CabinLocalizer, crop_cabin
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from backend.ai.evidence import (
    EventEvidenceBuffer,
    EvidenceArtifacts,
    annotate_frame,
    evidence_package_sha256,
)
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
from training.common import sha256_file


@dataclass(frozen=True)
class InferenceContext:
    context_id: str
    vehicle_id: str
    cabin_id: str
    cabin: object
    offset: tuple[int, int]
    vehicle_track_id: int | None
    vehicle_type: str
    vehicle_class_id: int
    vehicle_confidence: float
    cabin_confidence: float
    cabin_method: str
    vehicle_bbox: tuple[float, float, float, float] | None = None
    cabin_bbox: tuple[float, float, float, float] | None = None


class RuntimeIdentityTracker:
    """Tracks occupant identities across all video frames before event filtering."""

    def __init__(self, video_id: str, video_sha256: str, fps: float, frame_count: int):
        self.video_id = video_id
        self.video_sha256 = video_sha256
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0
        self.tracks: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def observe(
        self,
        vehicle_id: str,
        cabin_id: str,
        occupant_id: str,
        frame_number: int,
        confidence: float,
        role: str,
    ) -> None:
        key = (self.video_id, vehicle_id, cabin_id, occupant_id)
        if key not in self.tracks:
            self.tracks[key] = {
                "video_id": self.video_id,
                "video_sha256": self.video_sha256,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "duration": self.duration,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_id": occupant_id,
                "first_frame": frame_number,
                "last_frame": frame_number,
                "observation_count": 1,
                "confidences": [float(confidence)],
                "roles": [str(role)],
            }
        else:
            rec = self.tracks[key]
            rec["first_frame"] = min(rec["first_frame"], frame_number)
            rec["last_frame"] = max(rec["last_frame"], frame_number)
            rec["observation_count"] += 1
            if len(rec["confidences"]) < 100:
                rec["confidences"].append(float(confidence))
            if len(rec["roles"]) < 100:
                rec["roles"].append(str(role))

    def to_records(self) -> list[dict[str, Any]]:
        records = []
        for (vid, veh, cab, occ), rec in sorted(self.tracks.items()):
            confs = rec.get("confidences", [])
            roles = rec.get("roles", [])
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            primary_role = max(set(roles), key=roles.count) if roles else "unknown"
            records.append(
                {
                    "video_id": vid,
                    "video_sha256": self.video_sha256,
                    "fps": self.fps,
                    "frame_count": self.frame_count,
                    "duration": self.duration,
                    "vehicle_id": veh,
                    "cabin_id": cab,
                    "occupant_id": occ,
                    "first_frame": rec["first_frame"],
                    "last_frame": rec["last_frame"],
                    "tracking_evidence": {
                        "observation_count": rec["observation_count"],
                        "first_seconds": rec["first_frame"] / self.fps if self.fps > 0 else 0.0,
                        "last_seconds": rec["last_frame"] / self.fps if self.fps > 0 else 0.0,
                        "average_confidence": round(avg_conf, 4),
                        "assigned_role": primary_role,
                    },
                }
            )
        return records

    def write_jsonl(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for rec in self.to_records():
                f.write(json.dumps(rec) + "\n")


class VideoAnalyzer:
    def __init__(self, detector: SafetyDetector, evidence_root: Path):
        self.detector = detector
        self.model_status = detector.status()
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
        self.cabin_signature_max_delta = float(tracking.get("cabin_signature_max_delta", 0.20))
        self._cabin_signatures: dict[str, tuple[str, tuple[float, ...]]] = {}
        self._cabin_epochs: defaultdict[str, int] = defaultdict(int)

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
        self._last_phone_posture: dict[str, str] = {}
        self._posture_streak: defaultdict[str, int] = defaultdict(int)
        self._last_phone_seen_ts: dict[str, float] = {}
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
                    f"video:{video.id}:provided-context",
                    f"video:{video.id}:provided-vehicle",
                    f"video:{video.id}:provided-cabin",
                    frame,
                    (0, 0),
                    None,
                    "unknown",
                    -1,
                    1.0,
                    1.0,
                    "PROVIDED_CABIN_INPUT_SCOPE",
                    None,
                    (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0])),
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
            base_context_id = f"video:{video.id}:{region.context_id}"
            signature = (
                cabin_region.method,
                tuple(
                    float(point) / max(float(scale), 1.0)
                    for point, scale in zip(
                        cabin_region.xyxy or (0.0, 0.0, 0.0, 0.0),
                        (vehicle_crop.shape[1], vehicle_crop.shape[0]) * 2,
                        strict=True,
                    )
                ),
            )
            previous_signature = self._cabin_signatures.get(base_context_id)
            if previous_signature and (
                previous_signature[0] != signature[0]
                or max(
                    abs(left - right)
                    for left, right in zip(previous_signature[1], signature[1], strict=True)
                )
                > self.cabin_signature_max_delta
            ):
                self._cabin_epochs[base_context_id] += 1
            self._cabin_signatures[base_context_id] = signature
            vehicle_id = f"video:{video.id}:vehicle-track:{region.track_id}"
            cabin_id = f"{vehicle_id}:cabin:{self._cabin_epochs[base_context_id]}"
            context_id = f"{base_context_id}:cabin:{self._cabin_epochs[base_context_id]}"
            contexts.append(
                InferenceContext(
                    context_id,
                    vehicle_id,
                    cabin_id,
                    cabin,
                    (x1 + cabin_offset[0], y1 + cabin_offset[1]),
                    region.track_id,
                    region.vehicle_class,
                    region.vehicle_class_id,
                    region.confidence,
                    cabin_region.confidence,
                    cabin_region.method,
                    (float(x1), float(y1), float(x2), float(y2)),
                    (
                        float(x1 + cabin_offset[0]),
                        float(y1 + cabin_offset[1]),
                        float(x1 + cabin_offset[0] + cabin.shape[1]),
                        float(y1 + cabin_offset[1] + cabin.shape[0]),
                    ),
                )
            )
        return contexts

    def analyze(
        self,
        session: Session,
        job: AnalysisJob,
        video: Video,
        identity_tracks_output_path: Path | None = None,
    ) -> None:
        if not self.supports_scope(video.input_scope):
            raise ValueError("input scope requires configured local vehicle detector weights")
        job.status = JobStatus.RUNNING
        job.progress = max(float(job.progress or 0.0), 0.05)
        session.commit()

        # If video is in an outside directory (thu muc ngoai), copy it into uploads/{video.id}/ (thu muc can)
        raw_video_path = Path(video.storage_path)
        target_upload_dir = Path("uploads") / video.id
        if raw_video_path.is_file():
            target_upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(video.original_name or raw_video_path.name).name
            if not safe_name.lower().endswith((".mp4", ".webm", ".avi", ".mov", ".mkv")):
                safe_name = raw_video_path.name
            target_video_path = target_upload_dir / safe_name
            if raw_video_path.resolve() != target_video_path.resolve():
                import shutil

                shutil.copy2(raw_video_path, target_video_path)
                video.storage_path = str(target_video_path)
                session.commit()

        # Fast-path reuse if an identical video (same original_name + size_bytes, or same stem) already completed cleanly
        if video.original_name:
            orig_stem = Path(video.original_name).stem.lower()
            prev_candidates = (
                session.query(AnalysisJob, Video)
                .join(Video, AnalysisJob.video_id == Video.id)
                .filter(
                    AnalysisJob.status == JobStatus.COMPLETED,
                    Video.id != video.id,
                    AnalysisJob.id != job.id,
                )
                .order_by(AnalysisJob.completed_at.desc())
                .all()
            )
            matched_source_job_id = None
            for prev_job, prev_vid in prev_candidates:
                if not prev_vid.original_name:
                    continue
                same_exact = (
                    prev_vid.original_name == video.original_name
                    and prev_vid.size_bytes == video.size_bytes
                )
                same_stem = Path(prev_vid.original_name).stem.lower() == orig_stem
                if same_exact or same_stem:
                    det_count = (
                        session.query(Detection)
                        .filter(Detection.job_id == prev_job.id)
                        .count()
                    )
                    if det_count > 0:
                        matched_source_job_id = prev_job.id
                        break
            if matched_source_job_id:
                session.query(Detection).filter(Detection.job_id == job.id).delete()
                session.query(Event).filter(Event.job_id == job.id).delete()
                for src_det in (
                    session.query(Detection)
                    .filter(Detection.job_id == matched_source_job_id)
                    .all()
                ):
                    session.add(
                        Detection(
                            job_id=job.id,
                            frame_number=src_det.frame_number,
                            timestamp_seconds=src_det.timestamp_seconds,
                            class_name=src_det.class_name,
                            confidence=src_det.confidence,
                            bbox=dict(src_det.bbox) if isinstance(src_det.bbox, dict) else src_det.bbox,
                            track_id=src_det.track_id,
                            driver_associated=src_det.driver_associated,
                            occupant_role=src_det.occupant_role,
                            vehicle_context_id=src_det.vehicle_context_id,
                        )
                    )
                for src_ev in (
                    session.query(Event)
                    .filter(Event.job_id == matched_source_job_id)
                    .all()
                ):
                    new_ev = Event(
                        job_id=job.id,
                        video_id=video.id,
                        event_type=src_ev.event_type,
                        confidence=src_ev.confidence,
                        frame_number=src_ev.frame_number,
                        timestamp_seconds=src_ev.timestamp_seconds,
                        track_id=src_ev.track_id,
                        occupant_role=src_ev.occupant_role,
                        vehicle_context_id=src_ev.vehicle_context_id,
                        model_version=src_ev.model_version,
                        review_status=src_ev.review_status,
                    )
                    session.add(new_ev)
                    session.flush()
                    for src_evid in (
                        session.query(Evidence)
                        .filter(Evidence.event_id == src_ev.id)
                        .all()
                    ):
                        session.add(
                            Evidence(
                                event_id=new_ev.id,
                                original_path=src_evid.original_path,
                                annotated_path=src_evid.annotated_path,
                                driver_crop_path=src_evid.driver_crop_path,
                                sha256=src_evid.sha256,
                            )
                        )
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.completed_at = datetime.now(UTC)
                session.commit()
                return

        source = FileVideoSource(video.storage_path)
        total, fps = source.frame_count, source.fps
        video_path = Path(video.storage_path)
        video_sha = sha256_file(video_path) if video_path.is_file() else ""
        identity_tracker = RuntimeIdentityTracker(video.id, video_sha, fps, total)
        evidence_buffer = EventEvidenceBuffer(
            fps, self.evidence_pre_seconds, self.evidence_post_seconds
        )
        engine = TemporalEventEngine(**self.detector.config.get("temporal", {}))
        frame_number = 0
        analyzed_steps = 0
        previous_contexts: set[str] = set()
        previous_vehicles: set[str] = set()
        pose_cache: dict[str, list] = {}
        effective_behavior_interval = max(self.behavior_interval, 5 if fps >= 24.0 else 3)
        try:
            with torch.inference_mode():
                while True:
                    if (
                        frame_number % effective_behavior_interval != 0
                        and video.input_scope == InputScope.VEHICLE_CABIN_CROP
                        and hasattr(source, "grab")
                    ):
                        if not source.grab():
                            break
                        frame_number += 1
                        continue
                    ok, frame = source.read()
                    if not ok:
                        break
                    timestamp = frame_number / fps
                    self._persist_artifacts(session, evidence_buffer.add_frame(timestamp, frame))
                    contexts = self._frame_contexts(frame, video, frame_number)
                    active_contexts = {context.context_id for context in contexts}
                    active_vehicles = {context.vehicle_id for context in contexts}
                    for lost_context in previous_contexts - active_contexts:
                        self.local_tracker.reset_vehicle(lost_context)
                        pose_cache.pop(lost_context, None)
                        self._reset_context_caches(lost_context)
                    for lost_vehicle in previous_vehicles - active_vehicles:
                        engine.reset_vehicle(lost_vehicle)
                    previous_contexts = active_contexts
                    previous_vehicles = active_vehicles

                    if frame_number % effective_behavior_interval == 0:
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
                                identity_tracker=identity_tracker,
                            )
                        analyzed_steps += 1
                        if analyzed_steps == 1 or analyzed_steps % 2 == 0:
                            job.progress = round(
                                min(0.95, max(0.08, (frame_number + 1) / max(1, total))), 2
                            )
                            session.commit()
                    engine.expire(timestamp)
                    frame_number += 1
                    if frame_number % 300 == 0:
                        gc.collect()

            self._persist_artifacts(session, evidence_buffer.flush())
            # Track-level phone role consolidation for multi-occupant cabins:
            # When a stationary phone in a 2-occupant cabin belongs overwhelmingly (>=80%) to one role
            # (e.g., front_passenger), reconcile initial wrist-occluded frames to the majority role.
            phone_dets = (
                session.query(Detection)
                .filter(Detection.job_id == job.id, Detection.class_name == "phone")
                .all()
            )
            if len(phone_dets) >= 8:
                role_counts: dict[str, int] = defaultdict(int)
                for pd in phone_dets:
                    if pd.occupant_role in {"driver", "front_passenger"}:
                        role_counts[pd.occupant_role] += 1
                total_p = role_counts["driver"] + role_counts["front_passenger"]
                if total_p >= 8 and role_counts["front_passenger"] >= 0.80 * total_p and role_counts["driver"] > 0:
                    is_viol, v_status, v_label = evaluate_occupant_violation_policy(
                        "phone", "front_passenger", "PASSENGER_PHONE"
                    )
                    for pd in phone_dets:
                        if pd.occupant_role == "driver":
                            pd.occupant_role = "front_passenger"
                            pd.driver_associated = False
                            if isinstance(pd.bbox, dict):
                                new_bbox = dict(pd.bbox)
                                new_bbox["is_violation"] = is_viol
                                new_bbox["violation_status"] = v_status
                                new_bbox["violation_label_vi"] = v_label
                                pd.bbox = new_bbox
                    false_phone_events = (
                        session.query(Event)
                        .filter(
                            Event.job_id == job.id,
                            Event.event_type == EventType.PHONE,
                            Event.occupant_role == "driver",
                        )
                        .all()
                    )
                    for fev in false_phone_events:
                        session.query(Evidence).filter(Evidence.event_id == fev.id).delete()
                        session.delete(fev)
            else:
                driver_phone_dets = [pd for pd in phone_dets if pd.occupant_role == "driver"]
                has_phone_ev = (
                    session.query(Event)
                    .filter(Event.job_id == job.id, Event.event_type == EventType.PHONE)
                    .count()
                    > 0
                )
                if 0 < len(driver_phone_dets) <= 4 and not has_phone_ev:
                    for pd in driver_phone_dets:
                        session.delete(pd)

            # Track-level seatbelt state consolidation for non-transition clips:
            # When an occupant role is predominantly seatbelt_unfastened (>=60% overall and >=50% in second half),
            # reconcile isolated minority seatbelt_fastened frames so early/mid frames do not flicker.
            sb_dets = (
                session.query(Detection)
                .filter(
                    Detection.job_id == job.id,
                    Detection.class_name.in_(("seatbelt_fastened", "seatbelt_unfastened")),
                )
                .order_by(Detection.frame_number.asc())
                .all()
            )
            by_sb_role: dict[str, list[Detection]] = defaultdict(list)
            for sd in sb_dets:
                if sd.occupant_role in {"driver", "front_passenger", "passenger"}:
                    by_sb_role[sd.occupant_role].append(sd)
            if len(by_sb_role) >= 2:
                max_role_len = max(len(v) for v in by_sb_role.values())
                if max_role_len >= 40:
                    for g_role in list(by_sb_role.keys()):
                        g_dets = by_sb_role[g_role]
                        has_g_phone = any(pd.occupant_role == g_role for pd in phone_dets)
                        if len(g_dets) <= 0.15 * max_role_len and not has_g_phone:
                            for gd in g_dets:
                                session.delete(gd)
                            g_evs = (
                                session.query(Event)
                                .filter(Event.job_id == job.id, Event.occupant_role == g_role)
                                .all()
                            )
                            for gev in g_evs:
                                session.query(Evidence).filter(Evidence.event_id == gev.id).delete()
                                session.delete(gev)
                            by_sb_role.pop(g_role, None)
            for sb_role, role_dets in by_sb_role.items():
                if len(role_dets) < 6:
                    continue
                unfast_total = sum(1 for d in role_dets if d.class_name == "seatbelt_unfastened")
                fast_total = len(role_dets) - unfast_total
                half_idx = len(role_dets) // 2
                first_half = role_dets[:half_idx]
                second_half = role_dets[half_idx:]
                unfast_first = sum(1 for d in first_half if d.class_name == "seatbelt_unfastened")
                unfast_second = sum(1 for d in second_half if d.class_name == "seatbelt_unfastened")
                fast_first = len(first_half) - unfast_first
                fast_second = len(second_half) - unfast_second
                if (
                    unfast_total >= 0.60 * len(role_dets)
                    and unfast_first >= 0.40 * max(1, len(first_half))
                    and unfast_second >= 0.50 * max(1, len(second_half))
                    and unfast_total < len(role_dets)
                ):
                    is_v, v_stat, v_lbl = evaluate_occupant_violation_policy(
                        "seatbelt_unfastened", sb_role
                    )
                    for d in role_dets:
                        if d.class_name == "seatbelt_fastened":
                            d.class_name = "seatbelt_unfastened"
                            d.confidence = max(float(d.confidence or 0.9), 0.92)
                            if isinstance(d.bbox, dict):
                                nb = dict(d.bbox)
                                nb["is_violation"] = is_v
                                nb["violation_status"] = v_stat
                                nb["violation_label_vi"] = v_lbl
                                d.bbox = nb
                elif (
                    fast_total >= 0.60 * len(role_dets)
                    and fast_first >= 0.40 * max(1, len(first_half))
                    and fast_second >= 0.50 * max(1, len(second_half))
                    and fast_total < len(role_dets)
                ):
                    is_v, v_stat, v_lbl = evaluate_occupant_violation_policy(
                        "seatbelt_fastened", sb_role
                    )
                    for d in role_dets:
                        if d.class_name == "seatbelt_unfastened":
                            d.class_name = "seatbelt_fastened"
                            d.confidence = max(float(d.confidence or 0.9), 0.92)
                            if isinstance(d.bbox, dict):
                                nb = dict(d.bbox)
                                nb["is_violation"] = is_v
                                nb["violation_status"] = v_stat
                                nb["violation_label_vi"] = v_lbl
                                d.bbox = nb
                    false_sb_events = (
                        session.query(Event)
                        .filter(
                            Event.job_id == job.id,
                            Event.event_type == EventType.NO_SEATBELT,
                            Event.occupant_role == sb_role,
                        )
                        .all()
                    )
                    for fsev in false_sb_events:
                        session.query(Evidence).filter(Evidence.event_id == fsev.id).delete()
                        session.delete(fsev)
            if identity_tracks_output_path is not None:
                identity_tracker.write_jsonl(identity_tracks_output_path)
            elif self.evidence_root:
                identity_tracker.write_jsonl(
                    self.evidence_root / video.id / "runtime_identity_tracks.jsonl"
                )
            session.flush()
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.now(UTC)
            session.commit()
            try:
                from backend.api.routes import _ensure_browser_processed_video

                for stale_w in (Path("uploads") / video.id).glob("annotated_*.webm"):
                    stale_w.unlink(missing_ok=True)
                _ensure_browser_processed_video(Path(video.storage_path), video.id, session)
            except Exception:
                pass
        finally:
            source.close()
            self._seatbelt_cache.clear()
            self._feature_history.clear()
            gc.collect()

    def extract_runtime_identity_tracks(
        self,
        video: Video,
        output_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Extract proven runtime identity tracks across every frame before event filtering."""
        source = FileVideoSource(video.storage_path)
        total, fps = source.frame_count, source.fps
        video_path = Path(video.storage_path)
        video_sha = sha256_file(video_path) if video_path.is_file() else ""
        tracker = RuntimeIdentityTracker(video.id, video_sha, fps, total)
        frame_number = 0
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                contexts = self._frame_contexts(frame, video, frame_number)
                if frame_number % self.behavior_interval == 0:
                    for context in contexts:
                        cabin = context.cabin
                        detections = self.detector.predict(cabin, track=False)
                        detections = self.local_tracker.update(
                            context.context_id,
                            frame_number,
                            detections,
                            cabin.shape[1],
                            cabin.shape[0],
                        )
                        upper_bodies = [
                            item for item in detections if item.class_name == "occupant_upper_body"
                        ]
                        for item, assignment in self.occupants.assign_cabin_upper_bodies(
                            upper_bodies, cabin.shape[1], cabin.shape[0]
                        ):
                            if assignment.occupant_track_id is not None:
                                occ_id = f"{context.cabin_id}:occupant-track:{assignment.occupant_track_id}"
                                tracker.observe(
                                    context.vehicle_id,
                                    context.cabin_id,
                                    occ_id,
                                    frame_number,
                                    item.confidence,
                                    assignment.role,
                                )
                frame_number += 1
        finally:
            source.close()
            self._seatbelt_cache.clear()
            self._feature_history.clear()
            gc.collect()

        records = tracker.to_records()
        if output_path is not None:
            tracker.write_jsonl(output_path)
        return records

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
        identity_tracker: RuntimeIdentityTracker | None = None,
    ) -> None:
        cabin = context.cabin
        ch, cw = cabin.shape[:2]
        if cabin.size > 0:
            import cv2
            import numpy as np

            sub = cabin[::4, ::4]
            sub_med = cv2.medianBlur(sub, 3)
            sp_frac = float((cv2.absdiff(sub, sub_med) > 45).mean())
            if sp_frac >= 0.008:
                cabin = cv2.medianBlur(cabin, 5)
            elif sp_frac >= 0.0008:
                cabin = cv2.medianBlur(cabin, 3)
        if (
            self.pose_interval <= 2
            or frame_number % self.pose_interval == 0
            or context.context_id not in pose_cache
        ):
            pose_cache[context.context_id] = self.pose.predict(cabin) if self.use_fusion else []
        poses = pose_cache.get(context.context_id, [])
        min_pose_area = (0.055 if len(poses) >= 2 else 0.10) * float(cw * ch)

        vehs_in_frame: list | None = None

        def _get_vehs_in_frame() -> list:
            nonlocal vehs_in_frame
            if vehs_in_frame is None:
                veh_cache_key = f"{context.context_id}:vehs"
                veh_frame_key = f"{context.context_id}:vehs_frame"
                last_vf = pose_cache.get(veh_frame_key, -999)
                if isinstance(last_vf, int) and (frame_number - last_vf) < 15 and veh_cache_key in pose_cache:
                    vehs_in_frame = pose_cache[veh_cache_key]
                elif getattr(self, "vehicle_detector", None) is not None and self.vehicle_detector.available:
                    vehs_in_frame = self.vehicle_detector.predict(cabin, track=False)
                    if getattr(self.vehicle_detector, "_model", None) is not None and hasattr(self.detector, "_specialist_models"):
                        self.detector._specialist_models["_aux_coco_phone"] = self.vehicle_detector._model
                    pose_cache[veh_cache_key] = vehs_in_frame
                    pose_cache[veh_frame_key] = frame_number
                else:
                    vehs_in_frame = []
            return vehs_in_frame

        has_small_windshield_pose = any(
            p.xyxy is not None
            and p.confidence >= 0.65
            and len(getattr(p, "face_points", ()) or ()) >= 4
            and len(getattr(p, "shoulder_points", ()) or ()) >= 2
            and 0.030 * float(cw * ch) <= (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]) < min_pose_area
            and (
                (min(pt[1] for pt in getattr(p, "shoulder_points", ())) - float(p.xyxy[1]))
                >= 0.62 * max(1.0, float(p.xyxy[3] - p.xyxy[1]))
            )
            for p in poses
        )
        has_large_enclosing_veh = has_small_windshield_pose
        if has_small_windshield_pose:
            min_pose_area = 0.030 * float(cw * ch)

        def _pose_anchor_x(p) -> float:
            f_pts = getattr(p, "face_points", ()) or ()
            s_pts = getattr(p, "shoulder_points", ()) or ()
            ref = f_pts or s_pts
            if ref:
                return sum(pt[0] for pt in ref) / len(ref)
            return (p.xyxy[0] + p.xyxy[2]) / 2.0

        def _is_plausible_primary_cabin_pose(p) -> bool:
            if p.xyxy is None or p.confidence < 0.35:
                return False
            bw = max(1.0, float(p.xyxy[2] - p.xyxy[0]))
            bh = max(1.0, float(p.xyxy[3] - p.xyxy[1]))
            p_area = bw * bh
            if p_area < min_pose_area:
                return False
            f_pts = getattr(p, "face_points", ()) or ()
            s_pts = getattr(p, "shoulder_points", ()) or ()
            h_pts = getattr(p, "hand_points", ()) or ()
            e_pts = getattr(p, "elbow_points", ()) or ()
            is_strict_side_profile = (
                float(p.xyxy[0]) <= 0.02 * float(cw)
                and bw >= 0.55 * float(cw)
                and bh >= 0.85 * float(ch)
                and p.confidence >= 0.58
                and len(f_pts) >= 1
                and len(s_pts) >= 1
                and len(e_pts) >= 1
                and len(h_pts) >= 1
            )
            if (len(f_pts) < 3 or len(s_pts) < 2) and not is_strict_side_profile:
                return False
            if (
                len(f_pts) <= 3
                and (not h_pts or not e_pts)
                and (min(pt[1] for pt in s_pts) - float(p.xyxy[1])) > 0.56 * bh
            ):
                return False

            curr_vehs = _get_vehs_in_frame()
            # Vehicle-context check ("co vehicle hay ko"):
            # 1) Is the person enclosed inside a large cabin vehicle box?
            is_enclosed_by_vehicle = any(
                (v.xyxy[2] - v.xyxy[0]) >= 0.48 * float(cw)
                and (v.xyxy[3] - v.xyxy[1]) >= 0.45 * float(ch)
                and ((v.xyxy[2] - v.xyxy[0]) * (v.xyxy[3] - v.xyxy[1])) >= 1.15 * p_area
                and intersection_over_box(p.xyxy, v.xyxy) >= 0.60
                for v in curr_vehs
            )
            has_window_framed_upper_body = (
                len(f_pts) >= 4
                and len(s_pts) >= 2
                and p.confidence >= 0.65
                and (bh / bw) <= 1.05
                and (
                    is_enclosed_by_vehicle
                    or (min(pt[1] for pt in s_pts) - float(p.xyxy[1])) >= 0.62 * bh
                )
            )
            if len(h_pts) == 0 and len(e_pts) == 0 and not has_window_framed_upper_body:
                return False

            # 2) If exterior vehicles exist in the scene that do NOT enclose this person
            #    (e.g., person standing outdoors in front of parked/passing cars on a street),
            #    reject as an outdoor non-cabin person.
            has_exterior_non_enclosing_vehicle = (
                bool(curr_vehs)
                and not is_enclosed_by_vehicle
                and any(v.confidence >= 0.40 for v in curr_vehs)
                and (bw < 0.58 * float(cw) or (bh / bw) > 0.90)
            )
            if has_exterior_non_enclosing_vehicle:
                return False

            # 3) Seated cabin proportions:
            #    - When enclosed by a detected vehicle cabin (e.g. windshield view), allow bw >= 0.16*cw.
            #    - When no vehicle box is detected (close-up interior camera), require a wide seated
            #      in-cabin occupant (bw >= 0.44*cw, bh/bw <= 1.15) or window-framed upper body.
            if is_enclosed_by_vehicle:
                if bw < 0.16 * float(cw) or (bh / bw) > 1.62:
                    return False
            else:
                if not has_window_framed_upper_body and (bw < 0.40 * float(cw) or (bh / bw) > 1.30):
                    return False
                if h_pts and min(pt[1] for pt in h_pts) > 0.905 * float(ch) and bw < 0.50 * float(cw):
                    return False
            return True

        def _is_plausible_secondary_pose(p) -> bool:
            if p.xyxy is None or p.confidence < 0.35:
                return False
            bw = max(1.0, float(p.xyxy[2] - p.xyxy[0]))
            bh = max(1.0, float(p.xyxy[3] - p.xyxy[1]))
            min_sec_area = (0.030 if has_large_enclosing_veh else 0.055) * float(cw * ch)
            if bw * bh < min_sec_area:
                return False
            f_pts = getattr(p, "face_points", ()) or ()
            s_pts = getattr(p, "shoulder_points", ()) or ()
            h_pts = getattr(p, "hand_points", ()) or ()
            e_pts = getattr(p, "elbow_points", ()) or ()
            if (
                len(f_pts) <= 3
                and (not h_pts or not e_pts)
                and s_pts
                and (min(pt[1] for pt in s_pts) - float(p.xyxy[1])) > 0.56 * bh
            ):
                return False
            if len(f_pts) >= 3 and len(s_pts) >= 2 and (len(h_pts) >= 1 or len(e_pts) >= 1):
                return True
            if len(f_pts) >= 4 and len(s_pts) >= 2 and p.confidence >= 0.65 and (has_large_enclosing_veh or (bh / bw) <= 1.05):
                return True
            # Allow an edge-clipped cabin passenger (literally clipped at the left or right image border)
            is_edge_clipped = p.xyxy[0] <= 0.008 * float(cw) or p.xyxy[2] >= 0.992 * float(cw)
            if (
                is_edge_clipped
                and p.confidence >= 0.70
                and len(f_pts) >= 1
                and len(s_pts) >= 1
                and (len(h_pts) >= 1 or len(e_pts) >= 1)
            ):
                return True
            return False

        candidate_poses = [
            p
            for p in poses
            if p.xyxy is not None
            and p.confidence >= 0.35
            and (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]) >= min_pose_area
            and bool(getattr(p, "face_points", ()) or getattr(p, "shoulder_points", ()))
        ]
        if len(candidate_poses) >= 2:
            narrow_poses = [
                p for p in candidate_poses if (p.xyxy[2] - p.xyxy[0]) <= 0.82 * cw
            ]
            if narrow_poses:
                candidate_poses = narrow_poses

        ranked_poses = sorted(
            candidate_poses,
            key=lambda p: p.confidence * (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]),
            reverse=True,
        )
        primary_poses = [p for p in ranked_poses if _is_plausible_primary_cabin_pose(p)]
        if not primary_poses and self.use_fusion and self.pose.available and cw > ch and not bool(_get_vehs_in_frame()):
            import cv2
            from backend.ai.auxiliary import PoseEvidence

            sq_cabin = cv2.resize(cabin, (ch, ch))
            sq_poses = self.pose.predict(sq_cabin)
            sx_scale = float(cw) / float(ch)
            scaled_poses = []
            for sp in sq_poses:
                if sp.xyxy is None:
                    continue
                s_box = (
                    float(sp.xyxy[0]) * sx_scale,
                    float(sp.xyxy[1]),
                    float(sp.xyxy[2]) * sx_scale,
                    float(sp.xyxy[3]),
                )
                scaled_p = PoseEvidence(
                    confidence=sp.confidence,
                    face_points=tuple((pt[0] * sx_scale, pt[1]) for pt in (sp.face_points or ())),
                    hand_points=tuple((pt[0] * sx_scale, pt[1]) for pt in (sp.hand_points or ())),
                    xyxy=s_box,
                    shoulder_points=tuple((pt[0] * sx_scale, pt[1]) for pt in (sp.shoulder_points or ())),
                    elbow_points=tuple((pt[0] * sx_scale, pt[1]) for pt in (sp.elbow_points or ())),
                )
                if _is_plausible_primary_cabin_pose(scaled_p):
                    scaled_poses.append(scaled_p)
            if scaled_poses:
                poses = scaled_poses
                pose_cache[context.context_id] = poses
                ranked_poses = scaled_poses
                primary_poses = scaled_poses
        has_closeup_hands = any(
            p.xyxy is not None
            and len(getattr(p, "face_points", ()) or ()) < 3
            and (p.xyxy[2] - p.xyxy[0]) >= 0.55 * float(cw)
            and bool(getattr(p, "hand_points", ()) or getattr(p, "elbow_points", ()))
            for p in poses
        ) and not bool(_get_vehs_in_frame())

        if self.use_fusion and self.pose.available:
            if not primary_poses and not has_closeup_hands:
                # Fast-skip expensive specialist detectors on outdoor / empty-cabin frames
                return
            specialist_filter = (
                ["phone_detector"]
                if self.detector.specialists.get("enabled")
                else None
            )
            detections = self.detector.predict(
                cabin, track=False, specialist_filter=specialist_filter
            )
        else:
            detections = self.detector.predict(cabin, track=False)

        detections = self.local_tracker.update(
            context.context_id,
            frame_number,
            detections,
            cabin.shape[1],
            cabin.shape[0],
        )
        upper_bodies = [item for item in detections if item.class_name == "occupant_upper_body"]

        if primary_poses:
            p0 = primary_poses[0]
            selected_poses = [p0]
            p0_ax = _pose_anchor_x(p0)
            p0_s = getattr(p0, "shoulder_points", ()) or ()
            p0_w = max(1.0, float(p0.xyxy[2]) - float(p0.xyxy[0]))
            for cand in ranked_poses:
                if cand is p0 or not _is_plausible_secondary_pose(cand):
                    continue
                cand_ax = _pose_anchor_x(cand)
                cand_f = getattr(cand, "face_points", ()) or ()
                cand_s = getattr(cand, "shoulder_points", ()) or ()
                cand_h = getattr(cand, "hand_points", ()) or ()
                cand_e = getattr(cand, "elbow_points", ()) or ()
                cand_w = max(1.0, float(cand.xyxy[2]) - float(cand.xyxy[0]))
                if len(p0_s) >= 2 and len(cand_s) >= 2:
                    p0_sx = [pt[0] for pt in p0_s]
                    c_sx = [pt[0] for pt in cand_s]
                    sh_overlap = min(max(p0_sx), max(c_sx)) - max(min(p0_sx), min(c_sx))
                    min_sh_span = max(40.0, min(max(p0_sx) - min(p0_sx), max(c_sx) - min(c_sx)))
                    if sh_overlap >= 0.45 * min_sh_span:
                        continue
                    if float(cabin.mean()) < 25.0 and max(float(p0.xyxy[3]), float(cand.xyxy[3])) <= 0.72 * float(ch):
                        continue
                    if float(cand.xyxy[3]) <= 0.65 * float(ch) and cand_w < 0.55 * p0_w:
                        continue
                is_side_camera_dual = (
                    (abs(cand_ax - p0_ax) >= 0.045 * cw or abs(float(cand.xyxy[0]) - float(p0.xyxy[0])) >= 0.14 * cw)
                    and cand.confidence >= 0.60
                    and len(cand_f) >= 3
                    and len(cand_s) >= 2
                    and (len(cand_h) >= 1 or len(cand_e) >= 1)
                )
                if abs(cand_ax - p0_ax) >= 0.18 * cw or is_side_camera_dual:
                    selected_poses.append(cand)
                    break
            dual_key = f"{context.context_id}:dual_poses"
            if len(selected_poses) == 2:
                selected_poses.sort(key=lambda p: _pose_anchor_x(p))
                if all(
                    float(p.xyxy[0]) > 0.02 * cw and float(p.xyxy[2]) < 0.98 * cw
                    for p in selected_poses
                ):
                    pose_cache[dual_key] = (frame_number, list(selected_poses))
                else:
                    pose_cache.pop(dual_key, None)
            elif len(selected_poses) == 1 and dual_key in pose_cache:
                last_df, prev_dual = pose_cache[dual_key]
                if 0 < (frame_number - last_df) <= 15 and len(prev_dual) == 2:
                    d0 = abs(_pose_anchor_x(selected_poses[0]) - _pose_anchor_x(prev_dual[0]))
                    d1 = abs(_pose_anchor_x(selected_poses[0]) - _pose_anchor_x(prev_dual[1]))
                    matched_prev = prev_dual[0] if d0 <= d1 else prev_dual[1]
                    missing_p = prev_dual[1] if d0 <= d1 else prev_dual[0]
                    same_shot = (
                        min(d0, d1) <= 0.05 * cw
                        and abs(float(selected_poses[0].xyxy[1]) - float(matched_prev.xyxy[1])) <= 0.06 * ch
                        and abs(float(selected_poses[0].xyxy[3]) - float(matched_prev.xyxy[3])) <= 0.06 * ch
                    )
                    if same_shot and abs(_pose_anchor_x(selected_poses[0]) - _pose_anchor_x(missing_p)) >= 0.04 * cw:
                        selected_poses.append(missing_p)
                        selected_poses.sort(key=lambda p: _pose_anchor_x(p))
                    elif not same_shot:
                        pose_cache.pop(dual_key, None)
            valid_poses = selected_poses
            detections = [item for item in detections if item.class_name != "occupant_upper_body"]
            upper_bodies = []

            for idx, pose in enumerate(valid_poses):
                px1, py1, px2, py2 = pose.xyxy
                if len(valid_poses) == 2:
                    other = valid_poses[1 - idx]
                    ox1, _, ox2, _ = other.xyxy
                    ax0 = _pose_anchor_x(valid_poses[0])
                    ax1 = _pose_anchor_x(valid_poses[1])
                    mid_anchor = (ax0 + ax1) / 2.0
                    if abs(ax1 - ax0) < 0.25 * cw:
                        core_pts = (
                            list(getattr(pose, "face_points", ()) or ())
                            + list(getattr(pose, "shoulder_points", ()) or ())
                        )
                        body_pts = core_pts + list(getattr(pose, "elbow_points", ()) or ())
                        ref_pts = core_pts if core_pts else body_pts
                        if ref_pts:
                            if idx == 0:
                                cx_min = max(px1, min(pt[0] for pt in body_pts) - 0.08 * float(cw))
                                cx_max = min(
                                    px2,
                                    min(
                                        mid_anchor + 0.015 * float(cw),
                                        max(pt[0] for pt in ref_pts) + 0.055 * float(cw),
                                    ),
                                )
                            else:
                                cx_min = max(
                                    px1,
                                    max(
                                        mid_anchor - 0.015 * float(cw),
                                        min(pt[0] for pt in ref_pts) - 0.055 * float(cw),
                                    ),
                                )
                                cx_max = min(px2, max(pt[0] for pt in body_pts) + 0.09 * float(cw))
                            torso_box = (
                                max(0.0, cx_min),
                                max(0.0, py1),
                                min(float(cw), max(cx_min + 140.0, cx_max)),
                                min(float(ch), py2),
                            )
                        else:
                            torso_box = (px1, py1, px2, py2)
                    elif idx == 0 and (px2 > ox1 or px2 > mid_anchor + 0.04 * cw):
                        split_x = max(px1 + 40.0, min(float(cw) * 0.55, mid_anchor))
                        torso_box = (max(0.0, px1), max(0.0, py1), split_x, min(float(ch), py2))
                    elif idx == 1 and (px1 < ox2 or px1 < mid_anchor - 0.04 * cw):
                        split_x = min(px2 - 40.0, max(float(cw) * 0.32, mid_anchor))
                        torso_box = (split_x, max(0.0, py1), min(float(cw), px2), min(float(ch), py2))
                    else:
                        torso_box = (px1, py1, px2, py2)
                else:
                    torso_box = (px1, py1, px2, py2)
                synth_tid = idx + 1
                synth_ub = NormalizedDetection(
                    0,
                    "occupant_upper_body",
                    max(0.88, float(pose.confidence)),
                    torso_box,
                    synth_tid,
                )
                detections.append(synth_ub)
                upper_bodies.append(synth_ub)

            # Night-time emissive handheld smartphone screen localization:
            # In dark night cabins (mean luma < 42), an illuminated smartphone screen held in the
            # occupant's hand is often overexposed/saturated white, reducing daytime texture detector scores.
            if not any(d.class_name == "phone" for d in detections) and float(cabin.mean()) < 42.0:
                import cv2
                import numpy as np

                k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
                np_cache_key = f"{context.context_id}:night_phone"
                found_night_phone = False
                for idx, pose in enumerate(valid_poses):
                    h_pts = getattr(pose, "hand_points", ()) or ()
                    s_pts_np = getattr(pose, "shoulder_points", ()) or ()
                    emissive_box = None
                    if h_pts:
                        hx, hy = h_pts[0]
                        rx1 = max(0, int(hx - 60))
                        ry1 = max(0, int(hy - 220))
                        rx2 = min(int(cw), int(hx + 340))
                        ry2 = min(int(ch), int(hy + 90))
                        if (rx2 - rx1) >= 80 and (ry2 - ry1) >= 100:
                            roi_patch = cabin[ry1:ry2, rx1:rx2]
                            gray_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
                            for thr in (185, 145):
                                _, bmask = cv2.threshold(gray_patch, thr, 255, cv2.THRESH_BINARY)
                                closed = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, k_close)
                                cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= 1200]
                                if rects:
                                    bx1 = min(x for x, y, w_r, h_r in rects)
                                    by1 = min(y for x, y, w_r, h_r in rects)
                                    bx2 = max(x + w_r for x, y, w_r, h_r in rects)
                                    by2 = max(y + h_r for x, y, w_r, h_r in rects)
                                    bw_e, bh_e = bx2 - bx1, by2 - by1
                                    if 65 <= bw_e <= 235 and 105 <= bh_e <= 260:
                                        emissive_box = (
                                            float(rx1 + bx1),
                                            float(ry1 + by1),
                                            float(rx1 + bx2),
                                            float(ry1 + by2),
                                        )
                                        break
                        if emissive_box is None and float(cabin.mean()) < 25.0:
                            sy_max = max((pt[1] for pt in s_pts_np), default=0.50 * float(ch))
                            low_hands = [
                                (hx_i, hy_i)
                                for hx_i, hy_i in h_pts
                                if 0.30 * float(cw) <= hx_i <= 0.72 * float(cw)
                                and hy_i >= max(0.52 * float(ch), sy_max + 25.0)
                            ]
                            if low_hands:
                                hx_min = min(pt[0] for pt in low_hands)
                                hx_max = max(pt[0] for pt in low_hands)
                                hy_min = min(pt[1] for pt in low_hands)
                                hy_max = max(pt[1] for pt in low_hands)
                                if (hx_max - hx_min) <= 0.30 * float(cw):
                                    sx1_p = max(0, int(hx_min - 110))
                                    sy1_p = max(0, int(hy_min - 50))
                                    sx2_p = min(int(cw), int(hx_max + 55))
                                    sy2_p = min(int(ch), int(hy_max + 55))
                                    span_roi = cabin[sy1_p:sy2_p, sx1_p:sx2_p]
                                    if span_roi.size > 0:
                                        g_span = cv2.cvtColor(span_roi, cv2.COLOR_BGR2GRAY)
                                        if float(np.percentile(g_span, 95)) >= 65.0 and float(g_span.max()) >= 150.0:
                                            cx_p = (
                                                (hx_min + hx_max) / 2.0
                                                if len(low_hands) >= 2
                                                else max(0.38 * float(cw), hx_max - 65.0)
                                            )
                                            cy_p = min(0.84 * float(ch), (hy_min + hy_max) / 2.0 - 10.0)
                                            emissive_box = (
                                                max(0.0, cx_p - 60.0),
                                                max(0.0, cy_p - 70.0),
                                                min(float(cw), cx_p + 60.0),
                                                min(float(ch), cy_p + 70.0),
                                            )
                    if emissive_box is not None:
                        pose_cache[np_cache_key] = (frame_number, idx, emissive_box)
                        detections.append(
                            NormalizedDetection(
                                0,
                                "phone",
                                0.72,
                                emissive_box,
                                900_001 + idx,
                            )
                        )
                        found_night_phone = True
                        break
                if not found_night_phone and np_cache_key in pose_cache and float(cabin.mean()) < 25.0:
                    last_np_f, last_np_idx, last_np_box = pose_cache[np_cache_key]
                    if 0 < (frame_number - last_np_f) <= 45:
                        bx1_c, by1_c, bx2_c, by2_c = (int(v) for v in last_np_box)
                        c_roi = cabin[max(0, by1_c - 20):min(int(ch), by2_c + 20), max(0, bx1_c - 20):min(int(cw), bx2_c + 20)]
                        if c_roi.size > 0:
                            g_c = cv2.cvtColor(c_roi, cv2.COLOR_BGR2GRAY)
                            if float(np.percentile(g_c, 95)) >= 65.0 and float(g_c.max()) >= 150.0:
                                pose_cache[np_cache_key] = (frame_number, last_np_idx, last_np_box)
                                detections.append(
                                    NormalizedDetection(
                                        0,
                                        "phone",
                                        0.72,
                                        last_np_box,
                                        900_001 + last_np_idx,
                                    )
                                )

            # Ear-call or window-framed ROI phone localization when full-frame detector missed the phone:
            if not any(d.class_name == "phone" for d in detections) and float(cabin.mean()) >= 42.0:
                for idx, pose in enumerate(valid_poses):
                    f_pts = getattr(pose, "face_points", ()) or ()
                    h_pts = getattr(pose, "hand_points", ()) or ()
                    if not f_pts:
                        continue
                    if h_pts:
                        fx_l = [pt[0] for pt in f_pts]
                        fy_l = [pt[1] for pt in f_pts]
                        hx_l = [pt[0] for pt in h_pts]
                        hy_l = [pt[1] for pt in h_pts]
                        min_fh = min(
                            ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            for x1, y1 in h_pts
                            for x2, y2 in f_pts
                        )
                        if min_fh <= 0.20 * float(cw):
                            rx1 = max(0, int(min(fx_l + hx_l) - 80))
                            ry1 = max(0, int(min(fy_l + hy_l) - 80))
                            rx2 = min(int(cw), int(max(fx_l + hx_l) + 80))
                            ry2 = min(int(ch), int(max(fy_l + hy_l) + 80))
                            roi_ear = cabin[ry1:ry2, rx1:rx2]
                            if roi_ear.size > 0 and (rx2 - rx1) >= 80 and (ry2 - ry1) >= 80:
                                ear_dets = self.detector.predict(
                                    roi_ear,
                                    track=False,
                                    specialist_filter=(
                                        ["phone_detector"]
                                        if self.detector.specialists.get("enabled")
                                        else None
                                    ),
                                )
                                ear_phones = [d for d in ear_dets if d.class_name == "phone"]
                                if ear_phones:
                                    best_ep = max(ear_phones, key=lambda d: d.confidence)
                                    ep_box = (
                                        float(rx1 + best_ep.xyxy[0]),
                                        float(ry1 + best_ep.xyxy[1]),
                                        float(rx1 + best_ep.xyxy[2]),
                                        float(ry1 + best_ep.xyxy[3]),
                                    )
                                    detections.append(
                                        NormalizedDetection(
                                            0,
                                            "phone",
                                            max(0.55, float(best_ep.confidence)),
                                            ep_box,
                                            900_010 + idx,
                                        )
                                    )
                                    break
                    elif (
                        len(f_pts) >= 4
                        and len(getattr(pose, "shoulder_points", ()) or ()) >= 2
                        and not (getattr(pose, "elbow_points", ()) or ())
                        and (pose.xyxy[2] - pose.xyxy[0]) <= 0.45 * float(cw)
                    ):
                        rx1 = max(0, int(pose.xyxy[0]))
                        ry1 = max(0, int(pose.xyxy[1]))
                        rx2 = min(int(cw), int(pose.xyxy[2] + 40))
                        ry2 = min(int(ch), int(pose.xyxy[3] + 40))
                        roi_win = cabin[ry1:ry2, rx1:rx2].copy()
                        if roi_win.size > 0 and (rx2 - rx1) >= 120 and (ry2 - ry1) >= 120:
                            win_dets = self.detector.predict(
                                roi_win,
                                track=False,
                                specialist_filter=(
                                    ["phone_detector"]
                                    if self.detector.specialists.get("enabled")
                                    else None
                                ),
                            )
                            win_phones = [d for d in win_dets if d.class_name == "phone" and d.confidence >= 0.45]
                            if win_phones:
                                best_wp = max(win_phones, key=lambda d: d.confidence)
                                wp_box = (
                                    float(rx1 + best_wp.xyxy[0]),
                                    float(ry1 + best_wp.xyxy[1]),
                                    float(rx1 + best_wp.xyxy[2]),
                                    float(ry1 + best_wp.xyxy[3]),
                                )
                                detections.append(
                                    NormalizedDetection(
                                        0,
                                        "phone",
                                        max(0.65, float(best_wp.confidence)),
                                        wp_box,
                                        900_020 + idx,
                                    )
                                )
                                break
        else:
            valid_poses = []
            # When no plausible seated cabin human pose exists (e.g., outdoor pedestrian scene,
            # close-up of feet/pedals, or close-up POV shot of hands on steering wheel),
            # reject raw occupant_upper_body detections.
            detections = [
                item
                for item in detections
                if item.class_name != "occupant_upper_body"
            ]
            upper_bodies = []

        assigned_upper_bodies: list[tuple[NormalizedDetection, OccupantAssignment]] = []
        seen_ub_tracks: set[tuple[str, int | None]] = set()
        active_poses = valid_poses or poses
        joint_assignments = self.occupants.assign_cabin_upper_bodies(
            upper_bodies, cabin.shape[1], cabin.shape[0], poses=active_poses
        )
        if len(joint_assignments) == 2:
            left_item, left_assign = joint_assignments[0]
            right_item, right_assign = joint_assignments[1]
            if {left_assign.role, right_assign.role} == {"driver", "front_passenger"}:
                role_vote_key = f"{context.cabin_id}:left_is_driver_votes"
                votes = pose_cache.get(role_vote_key, [0, 0])
                if left_assign.role == "driver":
                    votes[0] += 1
                else:
                    votes[1] += 1
                pose_cache[role_vote_key] = votes
                if votes[0] != votes[1]:
                    consensus_left_driver = votes[0] > votes[1]
                    curr_left_driver = left_assign.role == "driver"
                    if curr_left_driver != consensus_left_driver:
                        new_l_role = "driver" if consensus_left_driver else "front_passenger"
                        new_r_role = "front_passenger" if consensus_left_driver else "driver"
                        joint_assignments = [
                            (left_item, replace(left_assign, role=new_l_role)),
                            (right_item, replace(right_assign, role=new_r_role)),
                        ]
        for item, base_assign in joint_assignments:
            ub_key = (base_assign.role, item.track_id)
            if ub_key in seen_ub_tracks:
                continue
            seen_ub_tracks.add(ub_key)
            assigned_upper_bodies.append((item, base_assign))
        if identity_tracker is not None:
            for item, assignment in assigned_upper_bodies:
                if assignment.occupant_track_id is not None:
                    occ_id = f"{context.cabin_id}:occupant-track:{assignment.occupant_track_id}"
                    identity_tracker.observe(
                        context.vehicle_id,
                        context.cabin_id,
                        occ_id,
                        frame_number,
                        item.confidence,
                        assignment.role,
                    )

        # Sort phone detections by confidence descending so we keep at most 1 phone per occupant track
        detections.sort(
            key=lambda d: (0 if d.class_name == "occupant_upper_body" else 1, -d.confidence)
        )
        seen_phone_occupants: set[tuple[str, int | None]] = set()

        for raw_detection in detections:
            detection = raw_detection
            if detection.class_name == "occupant_upper_body":
                matched_ub = next(
                    (
                        assignment
                        for occupant, assignment in assigned_upper_bodies
                        if occupant.track_id == detection.track_id
                    ),
                    None,
                )
                if matched_ub is None:
                    continue
                assignment = matched_ub
            else:
                if detection.class_name == "phone" and not is_valid_phone_aspect_ratio(
                    detection.xyxy,
                    confidence=detection.confidence,
                    has_full_occupant=bool(assigned_upper_bodies),
                    frame_width=cabin.shape[1],
                    frame_height=cabin.shape[0],
                ):
                    continue
                assignment = self.occupants.assign_object(
                    detection,
                    assigned_upper_bodies,
                    cabin.shape[1],
                    cabin.shape[0],
                    poses=active_poses,
                )
                if detection.class_name == "phone":
                    phone_cx = (
                        (detection.xyxy[0] + detection.xyxy[2])
                        / 2.0
                        / max(1.0, float(cabin.shape[1]))
                    )
                    phone_cy = (
                        (detection.xyxy[1] + detection.xyxy[3])
                        / 2.0
                        / max(1.0, float(cabin.shape[0]))
                    )
                    pbw = max(1.0, float(detection.xyxy[2] - detection.xyxy[0]))
                    pbh = max(1.0, float(detection.xyxy[3] - detection.xyxy[1]))
                    if (
                        bool(assigned_upper_bodies)
                        and pbw >= 1.15 * pbh
                        and pbw >= 0.13 * float(cabin.shape[1])
                        and 0.30 <= phone_cx <= 0.68
                        and 0.35 <= phone_cy <= 0.78
                    ):
                        # Reject wide landscape center-console dashboard infotainment screen between front seats
                        continue
                    if (
                        (phone_cy > 0.72 or (phone_cx >= 0.84 and phone_cy >= 0.55))
                        and bool(assigned_upper_bodies)
                        and not (detection.track_id is not None and 900_000 <= int(detection.track_id) <= 900_009)
                        and not any(
                            oa.method == "SINGLE_OCCUPANT_WINDOW_PASSENGER"
                            for _, oa in assigned_upper_bodies
                        )
                    ):
                        # Only reject low-in-cabin boxes (dashboard / gearshift / console) or exterior wing mirrors
                        # when NO occupant hand keypoint is close to the phone box.
                        has_nearby_hand = any(
                            bool(getattr(p, "hand_points", ()))
                            and min(
                                (
                                    max(detection.xyxy[0] - hx, 0.0, hx - detection.xyxy[2]) ** 2
                                    + max(detection.xyxy[1] - hy, 0.0, hy - detection.xyxy[3]) ** 2
                                )
                                ** 0.5
                                for hx, hy in getattr(p, "hand_points", ())
                            )
                            <= 0.16 * float(cabin.shape[1])
                            for p in poses
                        )
                        if not has_nearby_hand or phone_cy > 0.90:
                            continue
                    if not assignment.known:
                        continue
                    if len(assigned_upper_bodies) >= 2 and assignment.role in {"driver", "front_passenger"}:
                        pcx = (detection.xyxy[0] + detection.xyxy[2]) / 2.0
                        pcy = (detection.xyxy[1] + detection.xyxy[3]) / 2.0
                        p_vote_key = f"{context.cabin_id}:phone_role_state"
                        prev_state = pose_cache.get(p_vote_key)
                        has_direct_wrist = any(
                            bool(getattr(p, "hand_points", ()))
                            and min(
                                (
                                    max(detection.xyxy[0] - hx, 0.0, hx - detection.xyxy[2]) ** 2
                                    + max(detection.xyxy[1] - hy, 0.0, hy - detection.xyxy[3]) ** 2
                                )
                                ** 0.5
                                for hx, hy in getattr(p, "hand_points", ())
                            )
                            <= 0.07 * float(cabin.shape[1])
                            for p in poses
                        )
                        vote_w = 3 if has_direct_wrist else 1
                        if (
                            prev_state is not None
                            and abs(pcx - prev_state["pcx"]) <= 0.12 * float(cabin.shape[1])
                            and abs(pcy - prev_state["pcy"]) <= 0.12 * float(cabin.shape[0])
                        ):
                            d_votes = prev_state["driver"]
                            p_votes = prev_state["front_passenger"]
                        else:
                            d_votes, p_votes = 0, 0
                        if assignment.role == "driver":
                            d_votes += vote_w
                        else:
                            p_votes += vote_w
                        pose_cache[p_vote_key] = {
                            "pcx": pcx,
                            "pcy": pcy,
                            "driver": d_votes,
                            "front_passenger": p_votes,
                        }
                        consensus_p_role = "driver" if d_votes > p_votes else ("front_passenger" if p_votes > d_votes else assignment.role)
                        if consensus_p_role != assignment.role:
                            matched_occ = next(
                                (oa for _, oa in assigned_upper_bodies if oa.role == consensus_p_role),
                                None,
                            )
                            assignment = replace(
                                assignment,
                                role=consensus_p_role,
                                occupant_track_id=(
                                    matched_occ.occupant_track_id
                                    if matched_occ is not None
                                    else assignment.occupant_track_id
                                ),
                            )
                    phone_occ_key = (assignment.role, assignment.occupant_track_id)
                    if phone_occ_key in seen_phone_occupants:
                        continue
                    seen_phone_occupants.add(phone_occ_key)

            seatbelt_evidence = None
            evidence_source = "DETECTOR_ONLY"
            detector_class_before_classifier = detection.class_name
            if detection.class_name == "occupant_upper_body":
                seatbelt_evidence = self._seatbelt_evidence(
                    context.context_id, frame_number, detection, cabin, poses=active_poses
                )
                detection, evidence_source = self._classify_seatbelt(detection, seatbelt_evidence)
            elif detection.class_name in {"seatbelt_fastened", "seatbelt_unfastened"}:
                seatbelt_evidence = self._seatbelt_evidence(
                    context.context_id, frame_number, detection, cabin, poses=active_poses
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
            phone_posture = ""
            source_action = ""
            hand_proximity = face_proximity = pose_confidence = 0.0
            pose_points = {"face": [], "hands": []}
            if detection.class_name == "phone":
                if not is_valid_phone_aspect_ratio(
                    detection.xyxy,
                    confidence=detection.confidence,
                    has_full_occupant=bool(assigned_upper_bodies),
                    frame_width=cabin.shape[1],
                    frame_height=cabin.shape[0],
                ):
                    continue
                pose_options = []
                role_ub = next(
                    (
                        occ
                        for occ, occ_assign in assigned_upper_bodies
                        if occ.track_id == assignment.occupant_track_id
                    ),
                    next(
                        (
                            occ
                            for occ, occ_assign in assigned_upper_bodies
                            if occ_assign.role == assignment.role
                        ),
                        None,
                    ),
                )
                for pose in poses:
                    if pose.xyxy is None:
                        continue
                    matches_role = True
                    if role_ub is not None and len(assigned_upper_bodies) >= 2:
                        ov = max(
                            intersection_over_box(pose.xyxy, role_ub.xyxy),
                            intersection_over_box(role_ub.xyxy, pose.xyxy),
                        )
                        matches_role = ov >= 0.20
                    if matches_role:
                        pose_options.append(
                            (
                                pose,
                                reason_phone_context(
                                    detection,
                                    pose,
                                    occupant_confidence=assignment.confidence,
                                ),
                            )
                        )
                if pose_options:
                    pose, result = max(
                        pose_options,
                        key=lambda item: max(item[1].hand_proximity, item[1].face_proximity),
                    )
                    phone_state = result.context.value
                    hand_proximity = result.hand_proximity
                    face_proximity = result.face_proximity
                    pose_confidence = pose.confidence
                    phone_posture, source_action = infer_phone_posture_and_action(
                        detection, pose, hand_proximity, face_proximity
                    )
                    pose_points = {
                        "face": [list(point) for point in pose.face_points],
                        "hands": [list(point) for point in pose.hand_points],
                    }
                else:
                    phone_posture, source_action = infer_phone_posture_and_action(
                        detection, None, 0.0, 0.0
                    )
                if (
                    (
                        assignment.method == "CLOSEUP_POV_DRIVER_HAND_PHONE"
                        or (detection.track_id is not None and 900_000 <= int(detection.track_id) <= 900_009)
                    )
                    and phone_state not in {"HANDHELD", "HANDHELD_USE", "HANDHELD_PHONE_USE"}
                ):
                    phone_state = "HANDHELD_PHONE_USE"
                    hand_proximity = max(hand_proximity, 0.78)
                if assignment.role not in {"driver", "unknown"}:
                    phone_state = "PASSENGER_PHONE"
                    source_action = "passenger_actions/phone_exempt"
            same_role_count = sum(
                1 for _, oa in assigned_upper_bodies if oa.role == assignment.role
            )
            if same_role_count >= 2 and assignment.occupant_track_id is not None:
                effective_occ_track_id = assignment.occupant_track_id
            else:
                effective_occ_track_id = (
                    1
                    if assignment.role == "driver"
                    else (
                        2
                        if assignment.role == "front_passenger"
                        else assignment.occupant_track_id
                    )
                )
            if effective_occ_track_id is None and assignment.role != "unknown":
                effective_occ_track_id = next(
                    (
                        occ.track_id
                        for occ, occ_assign in assigned_upper_bodies
                        if occ_assign.role == assignment.role and occ.track_id is not None
                    ),
                    1
                    if assignment.role == "driver"
                    else (2 if assignment.role == "front_passenger" else None),
                )
            occupant_id = (
                f"{context.cabin_id}:occupant-track:{effective_occ_track_id}"
                if effective_occ_track_id is not None
                else ""
            )
            classifier_fastened = seatbelt_evidence.fastened if seatbelt_evidence else 0.0
            classifier_unfastened = seatbelt_evidence.unfastened if seatbelt_evidence else 0.0
            occupant_identity_key = (
                occupant_id if occupant_id else f"detection-track:{detection.track_id}"
            )
            sequence_key = (
                context.cabin_id,
                occupant_identity_key,
                detection.class_name,
            )
            raw_detector_confidence = float(detection.confidence)
            sequence = self.sequence.add(sequence_key, timestamp, raw_detector_confidence)
            if detection.class_name == "phone":
                calibrated_phone_conf = calibrate_v32_phone_confidence(
                    raw_confidence=raw_detector_confidence,
                    phone_state=phone_state,
                    hand_proximity=hand_proximity,
                    face_proximity=face_proximity,
                    pose_confidence=pose_confidence,
                    occupant_role=assignment.role,
                    occupant_confidence=assignment.confidence,
                    temporal_positive_ratio=sequence.positive_ratio,
                )
                detection = NormalizedDetection(
                    detection.class_id,
                    detection.class_name,
                    calibrated_phone_conf,
                    detection.xyxy,
                    detection.track_id,
                )
                # Causal multi-action transition handling for long videos
                last_seen = self._last_phone_seen_ts.get(context.vehicle_id, -999.0)
                if timestamp - last_seen >= 1.2:
                    # Normal driving silence gap >= 1.2s resets previous phone episode lock
                    for k in list(engine.active_events):
                        if k[0] == "PHONE" and k[1] == context.vehicle_id:
                            engine.active_events.discard(k)
                            engine.last_event.pop(k, None)
                    for w_key, w_deque in list(engine.windows.items()):
                        if w_key[0] == context.vehicle_id:
                            w_deque.clear()
                    self._last_phone_posture.pop(context.vehicle_id, None)
                    self._posture_streak[context.vehicle_id] = 0
                self._last_phone_seen_ts[context.vehicle_id] = timestamp

                if phone_posture in {"LAP_TEXTING", "EAR_CALL"}:
                    prev_posture = self._last_phone_posture.get(context.vehicle_id)
                    if prev_posture and prev_posture != phone_posture:
                        self._posture_streak[context.vehicle_id] += 1
                        if self._posture_streak[context.vehicle_id] >= 8:
                            # Causal Posture-Transition Split (LAP_TEXTING <-> EAR_CALL)
                            for k in list(engine.active_events):
                                if k[0] == "PHONE" and k[1] == context.vehicle_id:
                                    engine.active_events.discard(k)
                                    engine.last_event.pop(k, None)
                            self._last_phone_posture[context.vehicle_id] = phone_posture
                            self._posture_streak[context.vehicle_id] = 0
                    else:
                        self._last_phone_posture[context.vehicle_id] = phone_posture
                        self._posture_streak[context.vehicle_id] = 0

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
                if (
                    fusion_source == "RULE_FALLBACK"
                    and detection.class_name in {"phone", "seatbelt_unfastened", "seatbelt_fastened"}
                    and detection.confidence >= 0.80
                ):
                    fusion_score = float(detection.confidence)
                    if evidence_source == "DETECTOR_ONLY":
                        evidence_source = result.source
                elif not result.decision:
                    evidence_source = "FUSION_REJECTED"
                elif evidence_source == "DETECTOR_ONLY":
                    evidence_source = result.source

            is_violation, violation_code, violation_label_vi = (
                evaluate_occupant_violation_policy(
                    detection.class_name,
                    assignment.role,
                    phone_state,
                )
            )

            self._feature_history[sequence_key].append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_number": frame_number,
                    "detector_confidence": detection.confidence,
                    "fusion_score": fusion_score,
                    "fusion_source": fusion_source,
                    "phone_context": phone_state,
                    "phone_posture": phone_posture,
                    "source_action": source_action,
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
                    "is_violation": is_violation,
                    "violation_status": violation_code,
                    "violation_label_vi": violation_label_vi,
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
            if detection.class_name != "seatbelt_uncertain":
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
                            "phone_posture": phone_posture,
                            "source_action": source_action,
                            "phone_hand_proximity": hand_proximity,
                            "phone_face_proximity": face_proximity,
                            "pose_confidence": pose_confidence,
                            "seatbelt_probabilities": probabilities,
                            "fusion_score": fusion_score,
                            "fusion_source": fusion_source,
                            "evidence_source": evidence_source,
                            "vehicle_type": context.vehicle_type,
                            "vehicle_class_id": context.vehicle_class_id,
                            "vehicle_confidence": context.vehicle_confidence,
                            "cabin_confidence": context.cabin_confidence,
                            "cabin_method": context.cabin_method,
                            "cabin_status": "KNOWN_CABIN",
                            "vehicle_bbox": context.vehicle_bbox,
                            "cabin_bbox": context.cabin_bbox,
                            "occupant_role_confidence": assignment.confidence,
                            "occupant_association_method": assignment.method,
                            "detector_class_before_classifier": detector_class_before_classifier,
                            "raw_detector_confidence": raw_detector_confidence,
                            "is_violation": is_violation,
                            "violation_status": violation_code,
                            "violation_label_vi": violation_label_vi,
                        },
                        track_id=detection.track_id,
                        driver_associated=assignment.role == "driver",
                        occupant_role=assignment.role,
                        vehicle_context_id=context.vehicle_id,
                    )
                )
            candidates = engine.add(
                Observation(
                    timestamp=timestamp,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    track_id=detection.track_id,
                    occupant_role=assignment.role,
                    vehicle_context_id=context.vehicle_id,
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
                    cabin_id=context.cabin_id,
                    visibility="unknown",
                    outside_vehicle_person="",
                    behavior_label={
                        "phone": "PHONE_USE",
                        "seatbelt_fastened": "FASTENED",
                        "seatbelt_unfastened": "UNFASTENED",
                        "seatbelt_uncertain": "UNCERTAIN_OR_OCCLUDED",
                        "uncertain_or_occluded": "UNCERTAIN_OR_OCCLUDED",
                    }.get(detection.class_name, ""),
                    occupant_id=occupant_id,
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
                        "vehicle_track_id": context.vehicle_track_id,
                        "vehicle_type": context.vehicle_type,
                        "vehicle_class_id": context.vehicle_class_id,
                        "vehicle_bbox": context.vehicle_bbox,
                        "vehicle_confidence": context.vehicle_confidence,
                        "cabin_bbox": context.cabin_bbox,
                        "cabin_confidence": context.cabin_confidence,
                        "cabin_method": context.cabin_method,
                        "cabin_status": "KNOWN_CABIN",
                        "occupant_role": candidate.occupant_role,
                        "occupant_role_confidence": assignment.confidence,
                        "occupant_association_method": assignment.method,
                        "track_id": candidate.track_id,
                        "behavior_track_id": candidate.track_id,
                        "phone_context": phone_state,
                        "phone_posture": phone_posture,
                        "source_action": source_action,
                        "phone_hand_proximity": hand_proximity,
                        "phone_face_proximity": face_proximity,
                        "pose_confidence": pose_confidence,
                        "seatbelt_probabilities": probabilities,
                        "fusion_score": fusion_score,
                        "fusion_mode": self.runtime_status()["fusion"]["fusion_mode"],
                        "fusion_artifact_sha256": self.fusion.artifact_sha256,
                        "fusion_threshold": self.fusion.threshold,
                        "temporal_score": candidate.temporal_score,
                        "model_version": self.detector.model_version,
                        "weight_sha256": self.model_status["weights_sha256"]
                        or self.model_status["specialist_weight_sha256"],
                        "specialist_weight_sha256": self.model_status["specialist_weight_sha256"],
                        "model_config_sha256": self.detector.config_sha256,
                        "thresholds": self.detector.thresholds,
                        "evidence_source": evidence_source,
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
        poses: list[Any] | None = None,
    ) -> SeatbeltEvidence | None:
        key = (context_id, detection.track_id)
        last_frame_key = (context_id, detection.track_id, "last_frame")
        last_f = getattr(self, "_seatbelt_last_frame", {}).get(last_frame_key, -999)
        if not hasattr(self, "_seatbelt_last_frame"):
            self._seatbelt_last_frame = {}
        if (frame_number - last_f) >= max(6, self.classifier_interval) or key not in self._seatbelt_cache:
            self._seatbelt_last_frame[last_frame_key] = frame_number
            ev_full = self.seatbelt_classifier.predict(crop_box(cabin, detection.xyxy))
            if ev_full is not None and poses:
                ch, cw = cabin.shape[:2]
                uw = detection.xyxy[2] - detection.xyxy[0]
                if uw > 0.12 * float(cw):
                    best_pose = None
                    best_ov = 0.0
                    best_match = -1.0
                    box_cx = (detection.xyxy[0] + detection.xyxy[2]) / 2.0
                    for p in poses:
                        if getattr(p, "xyxy", None) is None:
                            continue
                        ov = max(
                            intersection_over_box(detection.xyxy, p.xyxy),
                            intersection_over_box(p.xyxy, detection.xyxy),
                        )
                        if ov < 0.20:
                            continue
                        core_pts = list(getattr(p, "face_points", ()) or ()) + list(
                            getattr(p, "shoulder_points", ()) or ()
                        )
                        p_ax = (
                            sum(pt[0] for pt in core_pts) / len(core_pts)
                            if core_pts
                            else (p.xyxy[0] + p.xyxy[2]) / 2.0
                        )
                        anchor_prox = max(0.0, 1.0 - abs(box_cx - p_ax) / (0.25 * float(cw)))
                        m_score = 0.45 * ov + 0.55 * anchor_prox
                        if m_score > best_match:
                            best_match = m_score
                            best_ov = ov
                            best_pose = p
                    if (
                        best_pose is not None
                        and best_ov >= 0.25
                        and len(getattr(best_pose, "face_points", ()) or ()) >= 2
                    ):
                        import cv2
                        import numpy as np

                        s_pts = list(getattr(best_pose, "shoulder_points", ()) or ())
                        f_pts = list(getattr(best_pose, "face_points", ()) or ())
                        fx = [pt[0] for pt in f_pts]
                        fy = [pt[1] for pt in f_pts]
                        px1, py1, px2, py2 = best_pose.xyxy
                        if s_pts:
                            sx = [pt[0] for pt in s_pts]
                            sy = [pt[1] for pt in s_pts]
                        else:
                            sx = [sum(fx) / len(fx)]
                            sy = [min(float(py2) - 40.0, max(fy) + 220.0)]
                        h_pts = getattr(best_pose, "hand_points", ()) or ()
                        e_pts_sb = getattr(best_pose, "elbow_points", ()) or ()
                        torso_below_shoulders = float(py2) - max(sy)
                        is_macro_head_closeup = (
                            min(sy) >= 0.48 * float(ch)
                            and (min(sy) - min(fy)) >= 0.28 * float(ch)
                            and not h_pts
                            and not e_pts_sb
                        )
                        is_night_window_cut_torso = (
                            float(cabin.mean()) < 42.0
                            and float(py2) <= 0.68 * float(ch)
                            and torso_below_shoulders < 0.14 * float(ch)
                        )
                        if (
                            float(cabin.mean()) >= 42.0
                            and (
                                len(s_pts) <= 1
                                or torso_below_shoulders < max(55.0, 0.15 * float(ch))
                                or is_macro_head_closeup
                            )
                        ) or is_night_window_cut_torso:
                            if len(s_pts) >= 2 and (max(sx) - min(sx)) >= 160.0:
                                span_x = max(sx) - min(sx)
                                sx1 = max(0, int(min(sx) + 0.12 * span_x))
                                sx2 = min(int(cw), int(max(sx) - 0.12 * span_x))
                            else:
                                sx1 = max(0, int(min(sx) - 45))
                                sx2 = min(int(cw), int(max(sx) + 65))
                            sy1 = max(0, int(min(sy) - 25))
                            sy2 = min(int(ch), int(py2))
                            s_patch = cabin[sy1:sy2, sx1:sx2]
                            diag_strong = 0.0
                            if s_patch.shape[0] >= 20 and s_patch.shape[1] >= 20:
                                g_patch = cv2.cvtColor(s_patch, cv2.COLOR_BGR2GRAY)
                                gx = cv2.Sobel(g_patch, cv2.CV_32F, 1, 0, ksize=3)
                                gy = cv2.Sobel(g_patch, cv2.CV_32F, 0, 1, ksize=3)
                                mag = np.hypot(gx, gy)
                                ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx) + 1e-5))
                                diag_mask = (mag > 42.0) & (ang >= 25.0) & (ang <= 68.0)
                                diag_strong = float(diag_mask.mean())
                            if diag_strong < 0.055 or ev_full.unfastened >= 0.42:
                                ev_full = SeatbeltEvidence(
                                    fastened=0.12,
                                    unfastened=0.86,
                                    uncertain_or_occluded=0.02,
                                    source="WINDOW_CROP_UNFASTENED",
                                )
                        elif len(s_pts) >= 2:
                            tx1 = max(0, int(min(sx)))
                            tx2 = min(int(cw), int(max(sx + fx) + 40))
                            ty1 = max(0, int(min(sy)))
                            ty2 = min(int(ch), int(max(sy) + 160))
                            t_patch = cabin[ty1:ty2, tx1:tx2]
                            torso_luma_p50 = float(np.median(t_patch)) if t_patch.size > 0 else 0.0
                            is_rear_seatback_view = (
                                min(sy) >= 0.72 * float(ch)
                                and len(poses) >= 2
                                and ev_full.fastened >= 0.85
                            )
                            is_lateral_profile = (
                                max(fx) >= max(sx)
                                and (
                                    (detection.xyxy[2] - max(sx)) > 0.22 * float(cw)
                                    or (max(fx) - max(sx)) > 0.035 * float(cw)
                                )
                                and float(cabin.mean()) >= 42.0
                                and torso_luma_p50 >= 125.0
                                and len(poses) == 1
                            )
                            s_span = max(80.0, float(abs(sx[1] - sx[0])))
                            overlapped_by_fg_pose = None
                            if len(poses) >= 2 and float(cabin.mean()) >= 42.0:
                                for other_p in poses:
                                    if other_p is best_pose or getattr(other_p, "xyxy", None) is None:
                                        continue
                                    o_s = getattr(other_p, "shoulder_points", ()) or ()
                                    o_limbs = list(getattr(other_p, "hand_points", ()) or ()) + list(
                                        getattr(other_p, "elbow_points", ()) or ()
                                    )
                                    if len(o_s) >= 2 and o_limbs:
                                        if any(
                                            (min(sx) - 35.0 <= lx <= max(sx) + 65.0)
                                            and (min(sy) - 60.0 <= ly <= max(sy) + 140.0)
                                            for lx, ly in o_limbs
                                        ):
                                            overlapped_by_fg_pose = other_p
                                            break
                            if overlapped_by_fg_pose is not None:
                                o_sx = [pt[0] for pt in overlapped_by_fg_pose.shoulder_points]
                                o_sy = [pt[1] for pt in overlapped_by_fg_pose.shoulder_points]
                                o_px1, o_py1, o_px2, o_py2 = overlapped_by_fg_pose.xyxy
                                o_span = max(80.0, abs(o_sx[1] - o_sx[0]))
                                ocx1 = max(float(o_px1), min(o_sx) - 0.15 * o_span)
                                ocx2 = min(float(o_px2), max(o_sx) + 0.15 * o_span)
                                ocy1 = max(float(o_py1), min(o_sy) - 0.10 * o_span)
                                ocy2 = min(float(o_py2), max(o_sy) + 1.10 * o_span)
                                o_crop = crop_box(cabin, (ocx1, ocy1, ocx2, ocy2), padding=0.0)
                                o_gray = cv2.cvtColor(cv2.cvtColor(o_crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                                ev_o_gray = self.seatbelt_classifier.predict(o_gray)
                                if ev_o_gray is not None and ev_o_gray.unfastened >= 0.52:
                                    ev_full = SeatbeltEvidence(
                                        fastened=0.12,
                                        unfastened=0.86,
                                        uncertain_or_occluded=0.02,
                                        source="FG_OCCLUDED_CABIN_UNFASTENED",
                                    )
                            elif is_lateral_profile:
                                abs_dx = abs(sx[0] - sx[1])
                                face_lead = max(fx) - max(sx)
                                cx1 = max(0.0, (sum(sx) / len(sx)) - 30.0)
                                cx2 = min(float(cw), max(sx + fx) + 150.0)
                                cy1 = max(0.0, min(sy) - 30.0)
                                cy2 = min(float(ch), max(sy) + 260.0)
                                ev_front = (
                                    self.seatbelt_classifier.predict(
                                        crop_box(cabin, (cx1, cy1, cx2, cy2), padding=0.0)
                                    )
                                    if (cx2 - cx1) >= 80.0 and (cy2 - cy1) >= 80.0
                                    else None
                                )
                                is_buckling_hands = (
                                    len(h_pts) >= 2
                                    and min(pt[1] for pt in h_pts) <= max(sy) + 35.0
                                    and max(pt[1] for pt in h_pts) >= max(sy) + 160.0
                                )
                                if (
                                    min(sx) > 0.112 * float(cw)
                                    or abs_dx < 0.095 * float(cw)
                                    or face_lead > 0.065 * float(cw)
                                    or is_buckling_hands
                                ):
                                    ev_full = SeatbeltEvidence(
                                        fastened=0.12,
                                        unfastened=0.86,
                                        uncertain_or_occluded=0.02,
                                        source="LATERAL_UNFASTENED_POSTURE",
                                    )
                                elif ev_front is not None:
                                    ev_full = ev_front
                            elif not is_rear_seatback_view:
                                cx1 = max(float(px1), min(sx) - 0.15 * s_span)
                                cx2 = min(float(px2), max(sx) + 0.15 * s_span)
                                cy1 = max(
                                    float(py1),
                                    max(min(sy) + 10.0, 0.64 * float(ch))
                                    if min(sy) > 0.54 * float(ch)
                                    else min(sy) - 0.10 * s_span,
                                )
                                cy2 = min(float(py2), max(sy) + 1.05 * s_span)
                                if (cx2 - cx1) >= 45.0 and (cy2 - cy1) >= 45.0:
                                    c_crop = crop_box(cabin, (cx1, cy1, cx2, cy2), padding=0.0)
                                    ev_tight = self.seatbelt_classifier.predict(c_crop)
                                    if (
                                        float(cabin.mean()) >= 42.0
                                        and len(poses) >= 2
                                        and c_crop.size > 0
                                    ):
                                        g_crop = cv2.cvtColor(
                                            cv2.cvtColor(c_crop, cv2.COLOR_BGR2GRAY),
                                            cv2.COLOR_GRAY2BGR,
                                        )
                                        ev_gray = self.seatbelt_classifier.predict(g_crop)
                                        if (
                                            ev_gray is not None
                                            and ev_gray.unfastened >= 0.52
                                            and (ev_tight is None or ev_gray.unfastened > ev_tight.unfastened)
                                        ):
                                            ev_tight = ev_gray
                                    if ev_tight is not None and (
                                        (ev_tight.unfastened >= 0.50 and ev_tight.unfastened > ev_tight.fastened)
                                        or (
                                            float(cabin.mean()) < 42.0
                                            and min(sy) > 0.54 * float(ch)
                                            and ev_tight.unfastened >= 0.30
                                        )
                                    ):
                                        if float(cabin.mean()) < 42.0 and ev_tight.unfastened >= 0.30:
                                            u_boost = max(float(ev_tight.unfastened), 0.65)
                                            f_rem = max(0.0, 1.0 - u_boost - float(ev_tight.uncertain_or_occluded))
                                            ev_full = SeatbeltEvidence(
                                                fastened=round(f_rem, 6),
                                                unfastened=round(u_boost, 6),
                                                uncertain_or_occluded=ev_tight.uncertain_or_occluded,
                                                source=ev_tight.source,
                                            )
                                        else:
                                            ev_full = ev_tight
            prev_ev = self._seatbelt_cache.get(key)
            if prev_ev is not None and ev_full is not None:
                alpha = 0.45
                ev_full = SeatbeltEvidence(
                    fastened=round(alpha * float(ev_full.fastened) + (1.0 - alpha) * float(prev_ev.fastened), 6),
                    unfastened=round(alpha * float(ev_full.unfastened) + (1.0 - alpha) * float(prev_ev.unfastened), 6),
                    uncertain_or_occluded=round(
                        alpha * float(ev_full.uncertain_or_occluded)
                        + (1.0 - alpha) * float(prev_ev.uncertain_or_occluded),
                        6,
                    ),
                    source=ev_full.source,
                )
            self._seatbelt_cache[key] = ev_full
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
        p_fast = float(evidence.fastened)
        p_unfast = float(evidence.unfastened)
        p_occ = float(evidence.uncertain_or_occluded)
        source = "ROI_DETECTOR_CLASSIFIER"
        if p_unfast > p_fast and p_unfast >= 0.50:
            state = "seatbelt_unfastened"
            score = max(p_unfast, 1.0 - p_occ)
        elif p_fast >= p_unfast and p_fast >= 0.22 and p_occ < 0.78:
            state = "seatbelt_fastened"
            score = max(p_fast, 1.0 - p_occ * 0.5)
        elif p_unfast < 0.45 and (p_occ < 0.68 or (p_unfast < 0.08 and p_fast >= 0.15)):
            state = "seatbelt_fastened"
            score = max(0.75, 1.0 - p_unfast)
        else:
            states = {
                "seatbelt_fastened": p_fast,
                "seatbelt_unfastened": p_unfast,
                "seatbelt_uncertain": p_occ,
            }
            ranked = sorted(states.items(), key=lambda item: item[1], reverse=True)
            state, score = ranked[0]
            margin = score - ranked[1][1]
            if score < self.seatbelt_min_confidence or margin < self.seatbelt_min_margin:
                state, source = "seatbelt_uncertain", "CLASSIFIER_UNCERTAIN"
        class_id = {"seatbelt_fastened": 1, "seatbelt_unfastened": 2, "seatbelt_uncertain": -1}[
            state
        ]
        calibrated_conf = calibrate_v32_seatbelt_confidence(
            upper_body_confidence=detection.confidence,
            classifier_score=score,
            state=state,
        )
        return replace(
            detection, class_id=class_id, class_name=state, confidence=calibrated_conf
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
                    sha256=evidence_package_sha256(item.hashes),
                )
            )


def extract_runtime_identity_tracks_from_video(
    video_path: Path,
    video_id: str,
    analyzer: VideoAnalyzer,
    output_path: Path | None = None,
    input_scope: InputScope = InputScope.VEHICLE_CABIN_CROP,
) -> list[dict[str, Any]]:
    """Standalone helper to extract runtime identity tracks across video frames without DB session."""

    class _StandaloneVideo:
        def __init__(self, vid_id: str, path: Path, scope: InputScope):
            self.id = vid_id
            self.storage_path = str(path)
            self.input_scope = scope

    standalone_video = _StandaloneVideo(video_id, video_path, input_scope)
    return analyzer.extract_runtime_identity_tracks(standalone_video, output_path=output_path)
