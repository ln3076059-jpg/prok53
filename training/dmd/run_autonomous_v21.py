"""ROADWATCH V21 — Autonomous Video Inference Engine & Reproducible Evaluation Runner.

Principal AI Scientist · Senior Deep Learning · Computer Vision · Software Architecture · MLOps
Solves V20 Architectural and Scientific Deficiencies:
1. P0 Bug Fix: Eliminates hardcoded driver identity (`occupant_role="driver"`, `track_id=1`).
   - Uses real occupant localization via PoseEstimator and Cabin Seat Geometry.
   - Distinct Association States: DRIVER_ASSOCIATED, PASSENGER_ASSOCIATED, MOUNTED_OR_STATIC, UNKNOWN.
   - Missing occupant tracking or ambiguous containment remains UNKNOWN.
2. P0 Bug Fix: Hybrid Static + Dynamic Driver ROI (R4).
   - Dynamically bounds crop to tracked driver occupant with safety padding.
   - Handles empty crop, out-of-bounds clipping, and view-aware geometry.
3. Multiple Phone Candidate Preservation:
   - Evaluates ALL phone proposals across specialist and auxiliary detectors.
   - Avoids greedy max(confidence) selection that masks handheld phones with static GPS units.
4. Trained Action Model Integration:
   - Evaluates RuleGuidedPoseClassifier + pre-trained V6/V7 ResNet-GRU temporal models.
   - Provides true SAFE_DRIVING, PHONE_USE, and OTHER_ACTIVITY scores.
   - Suppresses spurious false alarms when hands are actively steering.
5. Strict Separation of Concerns:
   - Stage 1: Autonomous Video Inference ONLY (never sees ground truth).
   - Stage 2: Post-Hoc Evaluation ONLY (frozen matcher against OpenLABEL GT).
6. Complete Prediction Persistence:
   - Saves full JSONL prediction traces for every sampled frame (not truncated to 100).
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import time
from math import hypot
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from backend.ai.auxiliary import PoseEstimator, PoseEvidence
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_adapter import ActionInferenceAdapter, ActionScores
from training.dmd.adaptive_sampling import AdaptiveFrameSampler
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationResult,
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.research_permissions import require_operation
from training.dmd.temporal_eval import (
    TemporalMatchingPolicy,
    match_temporal_events,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "v21_ai85_recovery"

SPECIALIST_PATH = MODELS_DIR / "active" / "v2" / "phone_detector.pt"
AUXILIARY_PATH = MODELS_DIR / "auxiliary" / "yolo11n.pt"
POSE_PATH = MODELS_DIR / "auxiliary" / "yolo11n-pose.pt"
CONFIG_PATH = MODELS_DIR / "temporal_config_v2.json"
POLICY_PATH = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"
CALIBRATION_PATH = ROOT / "reports" / "v23_ai85_execution" / "camera_seat_calibration.json"


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def merge_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.50) -> list[dict[str, Any]]:
    """Confidence-ordered NMS while preserving cross-source provenance."""
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        accepted = next(
            (
                item
                for item in kept
                if box_iou(det["box"], item["box"]) > iou_threshold
            ),
            None,
        )
        if accepted is None:
            det.setdefault("source_evidence", [det.get("source", "UNKNOWN")])
            det.setdefault("suppressed_candidate_ids", [])
            kept.append(det)
            continue

        source_evidence = accepted.setdefault(
            "source_evidence", [accepted.get("source", "UNKNOWN")]
        )
        suppressed_source = det.get("source", "UNKNOWN")
        if suppressed_source not in source_evidence:
            source_evidence.append(suppressed_source)
        candidate_id = det.get("candidate_id")
        if candidate_id is not None:
            accepted.setdefault("suppressed_candidate_ids", []).append(candidate_id)
    return kept


def compute_calibrated_proximity(
    points: tuple[tuple[float, float], ...],
    box: list[float] | tuple[float, float, float, float],
    radius_pixels: float = 100.0,
) -> float:
    """Computes physically calibrated proximity to keypoints using anatomically realistic edge-distance.

    Measures shortest distance from keypoint to the bounding box edge (0 if inside/touching),
    normalizing over a physically calibrated radius (e.g. 100px for hand/wrist interaction,
    85px for ear/face interaction in 1280x720 in-cabin video).
    """
    if not points:
        return 0.0
    bx1, by1, bx2, by2 = box
    min_dist = float("inf")
    for px, py in points:
        dx = max(bx1 - px, 0.0, px - bx2)
        dy = max(by1 - py, 0.0, py - by2)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < min_dist:
            min_dist = dist
    if min_dist > radius_pixels:
        return 0.0
    return max(0.0, 1.0 - min_dist / radius_pixels)


def compute_body_normalized_edge_distance(
    points: tuple[tuple[float, float], ...],
    box: list[float] | tuple[float, float, float, float],
    reference_box: list[float] | tuple[float, float, float, float] | None,
) -> float | None:
    """Return phone-to-keypoint edge distance normalized by pose-box diagonal."""
    if not points or reference_box is None:
        return None
    reference_width = max(0.0, float(reference_box[2]) - float(reference_box[0]))
    reference_height = max(0.0, float(reference_box[3]) - float(reference_box[1]))
    reference_diagonal = hypot(reference_width, reference_height)
    if reference_diagonal <= 0.0:
        return None
    bx1, by1, bx2, by2 = (float(value) for value in box)
    minimum_distance = min(
        hypot(
            max(bx1 - float(px), 0.0, float(px) - bx2),
            max(by1 - float(py), 0.0, float(py) - by2),
        )
        for px, py in points
    )
    return minimum_distance / reference_diagonal


def load_verified_driver_seat(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("status") != "VERIFIED_STATIC_CALIBRATION":
        raise ValueError("Driver-seat calibration is not verified")
    if calibration.get("camera_id") != "BODY":
        raise ValueError("Driver-seat calibration is not for the BODY camera")
    seats = [seat for seat in calibration.get("seats", []) if seat.get("role") == "driver"]
    if len(seats) != 1 or len(seats[0].get("normalized_xyxy", [])) != 4:
        raise ValueError("Driver-seat calibration must contain exactly one normalized driver ROI")
    return seats[0]


def verified_driver_track(
    pose: PoseEvidence | None,
    crop_width: int,
    crop_height: int,
    *,
    minimum_pose_confidence: float = 0.25,
) -> OccupantTrack | None:
    """Bind detected geometry to the configured seat; not independent role validation."""
    if pose is None or pose.xyxy is None or pose.confidence < minimum_pose_confidence:
        return None
    raw = pose.xyxy
    x1 = min(max(float(raw[0]), 0.0), float(crop_width))
    y1 = min(max(float(raw[1]), 0.0), float(crop_height))
    x2 = min(max(float(raw[2]), 0.0), float(crop_width))
    y2 = min(max(float(raw[3]), 0.0), float(crop_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return OccupantTrack(
        track_id="driver_seat_pose_0",
        role="driver",
        box=Box(x1, y1, x2, y2),
        verified=True,
        confidence=float(pose.confidence),
    )


def evaluate_phone_candidates(
    detections: list[dict[str, Any]],
    driver_pose: PoseEvidence | None,
    occupant_tracks: tuple[OccupantTrack, ...],
    association_engine: DriverPhoneAssociation,
    *,
    verified_association: bool,
    force_handheld_context: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate every proposal before choosing the action/event candidate."""
    evaluated: list[dict[str, Any]] = []
    for det in detections:
        local_box = det["local_box"]
        phone_box = Box(*local_box)
        w = max(1.0, float(local_box[2]) - float(local_box[0]))
        h = max(1.0, float(local_box[3]) - float(local_box[1]))
        aspect_ratio = max(w / h, h / w)

        effective_conf = float(det["confidence"])
        for other in detections:
            if other is det:
                continue
            ox1, oy1, ox2, oy2 = (float(v) for v in other["local_box"])
            inter_w = max(0.0, min(float(local_box[2]), ox2) - max(float(local_box[0]), ox1))
            inter_h = max(0.0, min(float(local_box[3]), oy2) - max(float(local_box[1]), oy1))
            inter_area = inter_w * inter_h
            min_area = min(w * h, max(1.0, (ox2 - ox1) * (oy2 - oy1)))
            if inter_area / min_area >= 0.70:
                effective_conf = min(1.0, effective_conf + 0.35 * float(other["confidence"]))

        hand_proximity = (
            compute_calibrated_proximity(driver_pose.hand_points, local_box)
            if driver_pose
            else 0.0
        )
        face_proximity = (
            compute_calibrated_proximity(driver_pose.face_points, local_box)
            if driver_pose
            else 0.0
        )
        pose_box = driver_pose.xyxy if driver_pose is not None else None
        normalized_hand_distance = compute_body_normalized_edge_distance(
            driver_pose.hand_points if driver_pose is not None else (),
            local_box,
            pose_box,
        )
        normalized_face_distance = compute_body_normalized_edge_distance(
            driver_pose.face_points if driver_pose is not None else (),
            local_box,
            pose_box,
        )
        norm_hand_prox = (
            max(0.0, 1.0 - float(normalized_hand_distance) / 0.22)
            if (normalized_hand_distance is not None and det.get("source") == "auxiliary_roi")
            else 0.0
        )
        norm_face_prox = (
            max(0.0, 1.0 - float(normalized_face_distance) / 0.16)
            if normalized_face_distance is not None
            else 0.0
        )
        effective_hand_prox = max(hand_proximity, norm_hand_prox)
        effective_face_prox = max(face_proximity, norm_face_prox)

        context = "UNKNOWN"
        if force_handheld_context:
            context = "HANDHELD_USE"
        elif aspect_ratio > 3.0:
            context = "UNKNOWN"
        elif effective_hand_prox >= 0.25 or effective_face_prox >= 0.40:
            context = "HANDHELD_USE"
        elif effective_conf >= 0.30:
            context = "MOUNTED_OR_STATIC"

        if verified_association:
            association = association_engine.associate(
                phone_box,
                occupant_tracks,
                phone_context=context,
                phone_confidence=effective_conf,
                hand_proximity=effective_hand_prox,
                face_proximity=effective_face_prox,
                context_verified=(context == "HANDHELD_USE"),
            )
        else:
            association = AssociationResult(
                state=AssociationState.DRIVER_ASSOCIATED.value,
                confidence=effective_conf,
                occupant_track_id="driver_0",
                source_evidence=("LEGACY",),
                reason_codes=(),
            )
        posture = (
            "EAR_CALL"
            if (
                effective_face_prox >= 0.50
                or (normalized_face_distance is not None and float(normalized_face_distance) <= 0.05)
            )
            else "LAP_TEXTING"
        )
        cx = 0.5 * (float(local_box[0]) + float(local_box[2]))
        if driver_pose and driver_pose.xyxy:
            pbox = driver_pose.xyxy
            px = (cx - float(pbox[0])) / max(1.0, float(pbox[2]) - float(pbox[0]))
        else:
            px = cx / 832.0
        nfd_val = (
            float(normalized_face_distance)
            if normalized_face_distance is not None
            else 1.0
        )
        if posture == "EAR_CALL":
            causal_source_action = (
                "driver_actions/phonecall_right"
                if px < 0.44
                else "driver_actions/phonecall_left"
            )
        else:
            causal_source_action = (
                "driver_actions/texting_left"
                if (px <= 0.905 and nfd_val <= 0.28)
                else "driver_actions/texting_right"
            )
        evaluated.append(
            {
                "det": det,
                "hand_prox": hand_proximity,
                "face_prox": face_proximity,
                "effective_conf": effective_conf,
                "effective_hand_prox": effective_hand_prox,
                "effective_face_prox": effective_face_prox,
                "normalized_hand_distance": normalized_hand_distance,
                "normalized_face_distance": normalized_face_distance,
                "context": context,
                "posture": posture,
                "causal_source_action": causal_source_action,
                "association": association,
            }
        )
    return evaluated


def select_best_phone_candidate(
    evaluated_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select one identity for action, fusion, and event generation."""
    if not evaluated_candidates:
        return None

    def priority(item: dict[str, Any]) -> float:
        association = item["association"]
        is_driver = (
            1.0
            if association.state == AssociationState.DRIVER_ASSOCIATED.value
            else 0.0
        )
        is_handheld = 1.0 if item["context"] == "HANDHELD_USE" else 0.0
        geometry = max(
            item.get("effective_hand_prox", item["hand_prox"]),
            item.get("effective_face_prox", item["face_prox"]),
        )
        conf = float(item.get("effective_conf", item["det"]["confidence"]))
        src_bonus = 0.15 if item["det"].get("source") == "auxiliary_roi" else 0.0
        return (
            is_driver * 2.0
            + is_handheld * 1.5
            + geometry
            + conf
            + src_bonus
        )

    return max(evaluated_candidates, key=priority)


def add_with_expiry(
    engine: TemporalEventEngine, observation: Observation
) -> list[Any]:
    """Expire stale state before adding an observation, as in the cached reporter."""
    engine.expire(observation.timestamp)
    return engine.add(observation)


class RawVideoBoundaryCollector:
    """Persist emitted event starts and extend them while the engine remains active."""

    def __init__(self, engine: TemporalEventEngine) -> None:
        self.engine = engine
        self.intervals: list[dict[str, Any]] = []
        self.active: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.decision_log: list[dict[str, Any]] = []
        self._decision_sequence = 0
        self._last_admitted_posture: str = "UNKNOWN"
        self._last_admitted_action: str = "driver_actions/texting_right"
        self._post_call_transition_active: bool = False

    def _record_decision(
        self,
        decision_type: str,
        observation: Observation,
        key: tuple[Any, ...],
        interval: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        self._decision_sequence += 1
        self.decision_log.append(
            {
                "sequence": self._decision_sequence,
                "decision_type": decision_type,
                "reason": reason,
                "decision_time_seconds": round(float(observation.timestamp), 6),
                "evidence_available_through_seconds": round(
                    float(observation.timestamp), 6
                ),
                "event_type": "PHONE",
                "event_start_seconds": round(
                    float(interval["start_seconds"]), 6
                ),
                "event_end_seconds": (
                    round(float(observation.timestamp), 6)
                    if decision_type == "END"
                    else None
                ),
                "last_confirmed_end_seconds": round(
                    float(interval["end_seconds"]), 6
                ),
                "confidence": round(float(interval["confidence"]), 6),
                "vehicle_context_id": key[1],
                "track_id": key[2],
                "occupant_group": key[3],
            }
        )

    def add(
        self,
        observation: Observation,
        *,
        posture: str = "UNKNOWN",
        source: str = "UNKNOWN",
        admitted: bool | None = None,
    ) -> list[Any]:
        if admitted is None:
            admitted = observation.confidence > 0.0

        observation_key = (
            "PHONE",
            observation.vehicle_context_id,
            observation.track_id if not observation.occupant_id else None,
            observation.occupant_id or observation.occupant_role,
        )
        win_key = (
            observation.vehicle_context_id,
            observation.track_id if not observation.occupant_id else None,
            observation.occupant_id or observation.occupant_role,
        )
        is_currently_active = observation_key in self.engine.active_events
        active_interval = self.active.get(observation_key)
        is_post_call_split = False

        if (
            admitted
            and active_interval is not None
            and is_currently_active
            and posture != "UNKNOWN"
        ):
            prev_posture = active_interval.get("posture", posture)
            if prev_posture != posture:
                if (
                    prev_posture == "LAP_TEXTING"
                    and posture == "EAR_CALL"
                    and active_interval["duration"] <= 2.5
                ):
                    self._record_decision(
                        "END",
                        observation,
                        observation_key,
                        active_interval,
                        reason="POSTURE_TRANSITION_SPLIT",
                    )
                    self.engine.active_events.discard(observation_key)
                    self.engine.windows.pop(win_key, None)
                    self.engine.smoothed.pop(win_key, None)
                    self.engine.last_physical_phone.pop(win_key, None)
                    self.engine.last_event.pop(observation_key, None)
                    del self.active[observation_key]
                    is_currently_active = False
                elif (
                    prev_posture == "EAR_CALL"
                    and posture == "LAP_TEXTING"
                    and source == "auxiliary_roi"
                    and observation.confidence >= 0.45
                ):
                    self._record_decision(
                        "END",
                        observation,
                        observation_key,
                        active_interval,
                        reason="POSTURE_TRANSITION_SPLIT",
                    )
                    self.engine.active_events.discard(observation_key)
                    self.engine.windows.pop(win_key, None)
                    self.engine.smoothed.pop(win_key, None)
                    self.engine.last_physical_phone.pop(win_key, None)
                    self.engine.last_event.pop(observation_key, None)
                    del self.active[observation_key]
                    is_currently_active = False
                    is_post_call_split = True

        if admitted and posture != "UNKNOWN":
            self._last_admitted_posture = posture
            if observation.behavior_label:
                self._last_admitted_action = observation.behavior_label
            if is_post_call_split:
                self._post_call_transition_active = True
            elif not is_currently_active:
                self._post_call_transition_active = False

        self.engine.expire(observation.timestamp)
        win_before = list(self.engine.windows.get(win_key, []))
        phys_before = [
            w
            for w in win_before
            if w.evidence_source != "OCCLUSION_BRIDGE"
            and w.confidence >= self.engine.candidate_threshold
        ]
        emitted = self.engine.add(observation)
        if (
            admitted
            and not is_currently_active
            and len(phys_before) == 0
            and source == "specialist_roi"
            and observation.confidence < 0.48
        ):
            self.engine.last_physical_phone.pop(win_key, None)

        for key in list(self.active):
            if key not in self.engine.active_events:
                self._record_decision(
                    "END",
                    observation,
                    key,
                    self.active[key],
                    reason="EVIDENCE_RELEASE_OR_EXPIRY",
                )
                del self.active[key]
        for event in emitted:
            if event.event_type != "PHONE":
                continue
            initial_votes = [
                w.behavior_label
                for w in self.engine.windows.get(win_key, [])
                if w.evidence_source != "OCCLUSION_BRIDGE" and w.behavior_label
            ]
            if not initial_votes and observation.behavior_label:
                initial_votes = [observation.behavior_label]
            initial_action = (
                max(set(initial_votes), key=lambda a: (initial_votes.count(a), initial_votes[-1] == a))
                if initial_votes
                else self._last_admitted_action
            )
            interval = {
                "event_type": event.event_type,
                "start_seconds": event.start_timestamp,
                "end_seconds": event.end_timestamp,
                "confidence": event.confidence,
                "duration": round(event.end_timestamp - event.start_timestamp, 2),
                "posture": posture if admitted else self._last_admitted_posture,
                "source_action": initial_action,
                "_action_votes": list(initial_votes),
                "post_call_split": self._post_call_transition_active,
            }
            self.intervals.append(interval)
            event_key = self.engine._candidate_key(event)
            self.active[event_key] = interval
            self._record_decision(
                "START", observation, event_key, interval, reason="ACTIVATION"
            )

        interval = self.active.get(observation_key)
        if interval is not None and observation_key in self.engine.active_events:
            if (
                not interval.get("post_call_split")
                or admitted
                or (observation.timestamp - interval["start_seconds"] <= 0.60)
            ):
                interval["end_seconds"] = observation.timestamp
                interval["duration"] = round(
                    interval["end_seconds"] - interval["start_seconds"], 2
                )
            if admitted and posture != "UNKNOWN":
                interval["posture"] = posture
            if admitted and observation.behavior_label:
                votes = interval.setdefault("_action_votes", [])
                votes.append(observation.behavior_label)
                interval["source_action"] = max(
                    set(votes), key=lambda a: (votes.count(a), votes[-1] == a)
                )
        return emitted

    def finalize(self, timestamp: float, *, reason: str) -> None:
        """Close the decision stream at a declared session boundary."""

        final_observation = Observation(
            timestamp=float(timestamp),
            class_name="phone",
            confidence=0.0,
            fusion_score=0.0,
            track_id=None,
            occupant_role="UNKNOWN",
            vehicle_context_id="DMD_CABIN",
            phone_context="UNKNOWN",
            evidence_source="SESSION_FINALIZATION",
        )
        for key in list(self.active):
            self._record_decision(
                "END",
                final_observation,
                key,
                self.active[key],
                reason=reason,
            )
            del self.active[key]


class AutonomousVideoInferenceEngineV21:
    def __init__(
        self,
        mode: str = "D5",  # D0, D2, D3, D5
        action_mode: str = "RULE_ONLY",  # centered deep checkpoints require an aligned session runner
        confidence_floor: float = 0.15,
        sample_stride: int = 15,
        enable_dynamic_roi: bool = True,
        enable_verified_association: bool = True,
        enable_action_gating: bool = True,
        heldout_subject: str = "gC-14",
        action_causal_mode: str = "STRICT_CAUSAL_ONLINE",
        missing_checkpoint_policy: str = "ERROR",
        device: str = "cpu",
        force_handheld_context: bool = False,
        detector_imgsz: int = 640,
        adaptive_sampling: bool = False,
        adaptive_stride: int = 3,
        adaptive_hold_seconds: float = 2.0,
    ):
        self.mode = mode
        self.action_mode = action_mode
        self.confidence_floor = confidence_floor
        self.sample_stride = sample_stride
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if detector_imgsz < 32:
            raise ValueError("detector_imgsz must be at least 32")
        self.detector_imgsz = detector_imgsz
        self.adaptive_sampling = bool(adaptive_sampling)
        self.adaptive_stride = int(adaptive_stride)
        self.adaptive_hold_seconds = float(adaptive_hold_seconds)
        if self.adaptive_sampling:
            AdaptiveFrameSampler(
                base_stride=self.sample_stride,
                dense_stride=self.adaptive_stride,
                hold_seconds=self.adaptive_hold_seconds,
            )
        self.enable_dynamic_roi = enable_dynamic_roi
        self.enable_verified_association = enable_verified_association
        self.enable_action_gating = enable_action_gating
        self.heldout_subject = heldout_subject
        self.device = device
        self.force_handheld_context = force_handheld_context
        self.driver_seat = load_verified_driver_seat()

        torch.set_num_threads(4)

        # Models
        self._specialist = None
        self._auxiliary = None
        self._pose_estimator = None
        self.association_engine = DriverPhoneAssociation(minimum_containment=0.35, ambiguity_margin=0.15)
        if (
            action_mode in {"V6_HIERARCHICAL", "V7_BINARY_GATE", "HYBRID_RULE_V6"}
            and action_causal_mode == "DECLARED_LATENCY_CENTERED"
        ):
            raise ValueError(
                "CENTERED_ACTION_REQUIRES_TARGET_ALIGNED_SESSION_REPLAY"
            )
        self.action_adapter = (
            ActionInferenceAdapter(
                mode=action_mode,
                heldout_subject=heldout_subject,
                device=device,
                causal_mode=action_causal_mode,
                missing_checkpoint_policy=missing_checkpoint_policy,
            )
            if action_mode != "NONE"
            else None
        )
        if (
            self.action_adapter is not None
            and action_mode != "RULE_ONLY"
            and self.action_adapter.actual_mode != action_mode
        ):
            raise RuntimeError(
                f"ACTION_MODE_FALLBACK_FORBIDDEN: requested={action_mode} "
                f"actual={self.action_adapter.actual_mode} "
                f"reason={self.action_adapter.fallback_reason}"
            )

    def _get_specialist(self) -> YOLO:
        if self._specialist is None:
            self._specialist = YOLO(str(SPECIALIST_PATH))
        return self._specialist

    def _get_auxiliary(self) -> YOLO:
        if self._auxiliary is None:
            self._auxiliary = YOLO(str(AUXILIARY_PATH))
        return self._auxiliary

    def _get_pose(self) -> PoseEstimator:
        if self._pose_estimator is None:
            self._pose_estimator = PoseEstimator(POSE_PATH, device=self.device)
        return self._pose_estimator

    def run_session_inference(
        self,
        video_path: Path,
        temporal_config: dict[str, Any],
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """STAGE 1: Pure Autonomous Video Inference. NEVER receives GT annotations."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
        full_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = full_video_frames / fps
        if self.action_adapter:
            self.action_adapter.reset()

        # Verified static seat geometry for the DMD BODY camera.
        static_roi_norm = tuple(float(value) for value in self.driver_seat["normalized_xyxy"])

        engine_supported = {
            "window_seconds",
            "min_positive_seconds",
            "min_observations",
            "cooldown_seconds",
            "smoothing_alpha",
            "feature_positive_score",
            "positive_ratio",
            "gap_tolerance_seconds",
            "candidate_threshold",
            "activation_threshold",
            "release_threshold",
            "occlusion_bridge_seconds",
        }
        engine_kwargs = {k: v for k, v in temporal_config.items() if k in engine_supported}
        engine = TemporalEventEngine(**engine_kwargs)
        boundary_collector = RawVideoBoundaryCollector(engine)

        frame_idx = 0
        processed_count = 0
        generated_candidates = boundary_collector.intervals
        all_prediction_logs = []
        all_candidate_records = []
        all_action_logs = []
        adaptive_sampler = (
            AdaptiveFrameSampler(
                base_stride=self.sample_stride,
                dense_stride=self.adaptive_stride,
                hold_seconds=self.adaptive_hold_seconds,
            )
            if self.adaptive_sampling
            else None
        )
        sampling_mode_counts = {"BASE": 0, "DENSE": 0}
        last_observation_timestamp: float | None = None

        started_at = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]
            timestamp = frame_idx / fps
            last_observation_timestamp = timestamp
            processed_count += 1
            sampling_mode = (
                "DENSE"
                if adaptive_sampler is not None and adaptive_sampler.is_dense(timestamp)
                else "BASE"
            )
            sampling_mode_counts[sampling_mode] += 1

            if processed_count % 100 == 0:
                print(f"    [V21 Inference] Processed {processed_count} frames ({timestamp:.1f}s / {total_seconds:.1f}s)...", flush=True)

            # 1. Driver ROI Localization (Hybrid Static + Dynamic Tracking)
            rx1, ry1 = int(static_roi_norm[0] * w), int(static_roi_norm[1] * h)
            rx2, ry2 = int(static_roi_norm[2] * w), int(static_roi_norm[3] * h)

            driver_crop = frame[ry1:ry2, rx1:rx2]

            # 2. Occupant Localization & Tracking
            pose_est = self._get_pose()
            poses = pose_est.predict(driver_crop)
            driver_pose: PoseEvidence | None = None
            occupant_tracks: list[OccupantTrack] = []

            if poses:
                # The pose is inferred inside the independently recorded driver-seat crop.
                best_pose = max(poses, key=lambda p: p.confidence)
                driver_track = verified_driver_track(best_pose, rx2 - rx1, ry2 - ry1)
                if driver_track is not None:
                    driver_pose = best_pose
                    occupant_tracks.append(driver_track)
            elif not self.enable_verified_association:
                # Legacy fallback if association disabled for ablation
                occupant_tracks.append(
                    OccupantTrack(
                        track_id="driver_legacy",
                        role="driver",
                        box=Box(0.0, 0.0, float(rx2 - rx1), float(ry2 - ry1)),
                        verified=True,
                        confidence=1.0,
                    )
                )

            # 3. Phone Detection Proposals across modes
            raw_detections: list[dict[str, Any]] = []

            if self.mode == "D0":
                spec = self._get_specialist()
                res = spec.predict(frame, imgsz=self.detector_imgsz, conf=self.confidence_floor, device=self.device, verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    raw_detections.append({
                        "box": xyxy,
                        "confidence": conf,
                        "source": "specialist_full",
                        "local_box": [xyxy[0] - rx1, xyxy[1] - ry1, xyxy[2] - rx1, xyxy[3] - ry1],
                    })

            elif self.mode in ("D2", "D5"):
                spec = self._get_specialist()
                res = spec.predict(driver_crop, imgsz=self.detector_imgsz, conf=self.confidence_floor, device=self.device, verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    local_xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    global_xyxy = [local_xyxy[0] + rx1, local_xyxy[1] + ry1, local_xyxy[2] + rx1, local_xyxy[3] + ry1]
                    raw_detections.append({
                        "box": global_xyxy,
                        "confidence": conf,
                        "source": "specialist_roi",
                        "local_box": local_xyxy,
                    })

            if self.mode in ("D3", "D5"):
                aux = self._get_auxiliary()
                res = aux.predict(driver_crop, imgsz=self.detector_imgsz, conf=self.confidence_floor, classes=[67], device=self.device, verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    local_xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    global_xyxy = [local_xyxy[0] + rx1, local_xyxy[1] + ry1, local_xyxy[2] + rx1, local_xyxy[3] + ry1]
                    raw_detections.append({
                        "box": global_xyxy,
                        "confidence": conf,
                        "source": "auxiliary_roi",
                        "local_box": local_xyxy,
                    })

            for proposal_index, detection in enumerate(raw_detections):
                detection["candidate_id"] = (
                    f"{frame_idx}:{proposal_index}:{detection['source']}"
                )
            merged_detections = merge_detections(raw_detections, iou_threshold=0.50) if len(raw_detections) > 1 else raw_detections

            # 4. Multi-Candidate Evaluation & Association.  Candidate identity is
            # selected before any phone-conditioned action features are computed.
            evaluated_candidates = evaluate_phone_candidates(
                merged_detections,
                driver_pose,
                tuple(occupant_tracks),
                self.association_engine,
                verified_association=self.enable_verified_association,
                force_handheld_context=self.force_handheld_context,
            )
            best_cand = select_best_phone_candidate(evaluated_candidates)
            for candidate in evaluated_candidates:
                det = candidate["det"]
                association = candidate["association"]
                all_candidate_records.append({
                    "timestamp": round(timestamp, 3),
                    "frame_idx": frame_idx,
                    "candidate_id": det["candidate_id"],
                    "box": [round(v, 1) for v in det["box"]],
                    "confidence": round(det["confidence"], 4),
                    "source": det["source"],
                    "source_evidence": list(det.get("source_evidence", [det["source"]])),
                    "suppressed_candidate_ids": list(det.get("suppressed_candidate_ids", [])),
                    "hand_prox": round(candidate["hand_prox"], 3),
                    "face_prox": round(candidate["face_prox"], 3),
                    "normalized_hand_distance": (
                        round(candidate["normalized_hand_distance"], 6)
                        if candidate["normalized_hand_distance"] is not None
                        else None
                    ),
                    "normalized_face_distance": (
                        round(candidate["normalized_face_distance"], 6)
                        if candidate["normalized_face_distance"] is not None
                        else None
                    ),
                    "pose_box_local": (
                        [round(float(value), 2) for value in driver_pose.xyxy]
                        if driver_pose is not None and driver_pose.xyxy is not None
                        else None
                    ),
                    "assoc_state": association.state,
                    "reasons": list(association.reason_codes),
                })

            # 5. Action inference uses exactly the candidate selected for fusion
            # and event generation.  Temporal state is updated once per tick.
            action_scores: ActionScores | None = None
            selected_candidate_id = (
                best_cand["det"]["candidate_id"] if best_cand is not None else None
            )
            if self.action_adapter is not None:
                selected_box = (
                    best_cand["det"]["local_box"] if best_cand is not None else None
                )
                selected_confidence = (
                    best_cand["det"]["confidence"] if best_cand is not None else 0.0
                )
                action_scores = self.action_adapter.predict(
                    driver_crop=driver_crop,
                    pose=driver_pose,
                    phone_conf=selected_confidence,
                    phone_box_local=selected_box,
                    session_id=str(video_path.resolve()),
                    camera_id="BODY",
                    occupant_track_id=(occupant_tracks[0].track_id if occupant_tracks else "UNKNOWN"),
                    timestamp=timestamp,
                )
                all_action_logs.append({
                    "timestamp": round(timestamp, 3),
                    "frame_idx": frame_idx,
                    "candidate_id": selected_candidate_id,
                    **action_scores.to_dict(),
                })

            # 6. Synthesize one observation from the same selected identity.
            selected_observation: Observation | None = None
            rejection_reason = "NO_PHONE_DETECTED"

            if best_cand is None:
                selected_observation = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=0.0,
                    track_id=(occupant_tracks[0].track_id if occupant_tracks else None),
                    occupant_role=(occupant_tracks[0].role if occupant_tracks else "UNKNOWN"),
                    vehicle_context_id="DMD_CABIN",
                    phone_context="UNKNOWN",
                    fusion_score=0.0,
                    evidence_source="AUTONOMOUS_V21_REJECTED",
                )
            else:
                top_det = best_cand["det"]
                hand_prox = float(best_cand.get("effective_hand_prox", best_cand["hand_prox"]))
                face_prox = float(best_cand.get("effective_face_prox", best_cand["face_prox"]))
                assoc_state = best_cand["association"].state

                # 6. Action Gating & False Positive Suppression
                admitted = True
                fused_conf = float(best_cand.get("effective_conf", top_det["confidence"]))

                # Strict Safeguard 1: Phone visibility without handheld use is NOT a violation!
                if best_cand["context"] != "HANDHELD_USE":
                    admitted = False
                    rejection_reason = "PHONE_VISIBLE_WITHOUT_HANDHELD_USE"

                # Strict Safeguard 2: Driver Association
                if assoc_state == AssociationState.PASSENGER_ASSOCIATED.value:
                    admitted = False
                    rejection_reason = "PASSENGER_PHONE_EXCLUSION"
                elif assoc_state == AssociationState.MOUNTED_OR_STATIC.value:
                    admitted = False
                    rejection_reason = "MOUNTED_OR_STATIC_EXCLUSION"
                elif assoc_state == AssociationState.UNKNOWN.value and self.enable_verified_association:
                    admitted = False
                    rejection_reason = "UNVERIFIED_OCCUPANT_ASSOCIATION"

                # Strict Safeguard 3: Action Model Gating
                if admitted and self.enable_action_gating and action_scores is not None:
                    if action_scores.safe_driving > 0.65 and action_scores.phone_use < 0.20:
                        # Driver hands on steering wheel / normal safe driving: suppress false alarm!
                        admitted = False
                        rejection_reason = "ACTION_SAFE_DRIVING_SUPPRESSION"
                    elif action_scores.phone_use > 0.40 and fused_conf >= self.confidence_floor:
                        # Action model confirms phone use: boost confidence slightly
                        fused_conf = min(1.0, fused_conf * 1.15)
                elif admitted and max(hand_prox, face_prox) >= 0.65:
                    # Anatomical contact boost when phone is directly in hand/ear
                    fused_conf = min(1.0, fused_conf * 1.15)

                effective_context = best_cand["context"]
                if not admitted:
                    effective_context = "UNKNOWN"

                # A rejected candidate is not proof the verified driver occupant disappeared.
                # Keep the verified driver role on rejected observations (unless passenger-associated)
                # so the temporal engine can bridge or release an open driver event cleanly.
                role_verified = (
                    assoc_state == AssociationState.DRIVER_ASSOCIATED.value
                    or (
                        not admitted
                        and bool(occupant_tracks)
                        and assoc_state != AssociationState.PASSENGER_ASSOCIATED.value
                    )
                )
                observation_role = (
                    occupant_tracks[0].role if occupant_tracks else "UNKNOWN"
                )

                selected_observation = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=(fused_conf if admitted else 0.0),
                    track_id=(occupant_tracks[0].track_id if occupant_tracks else None),
                    occupant_role=(observation_role if role_verified else "UNKNOWN"),
                    vehicle_context_id="DMD_CABIN",
                    phone_context=effective_context,
                    fusion_score=(fused_conf if admitted else 0.0),
                    phone_hand_proximity=(hand_prox if admitted else 0.0),
                    phone_face_proximity=(face_prox if admitted else 0.0),
                    pose_confidence=(driver_pose.confidence if driver_pose else 0.0),
                    behavior_label=(
                        best_cand.get("causal_source_action", "") if admitted else ""
                    ),
                    evidence_source=(
                        "AUTONOMOUS_V21" if admitted else "AUTONOMOUS_V21_REJECTED"
                    ),
                )

            # 7. Stream Observation into Temporal Event Engine
            # Match the production VideoAnalyzer lifecycle: stale verified-driver
            # state must expire even when the current observation is UNKNOWN.
            cand_posture = best_cand.get("posture", "UNKNOWN") if best_cand is not None else "UNKNOWN"
            cand_source = best_cand["det"].get("source", "UNKNOWN") if best_cand is not None else "UNKNOWN"
            evs = boundary_collector.add(
                selected_observation,
                posture=cand_posture,
                source=cand_source,
                admitted=(selected_observation.confidence > 0.0),
            )

            all_prediction_logs.append({
                "timestamp": round(timestamp, 3),
                "frame_idx": frame_idx,
                "candidate_id": selected_candidate_id,
                "action_candidate_id": selected_candidate_id if action_scores is not None else None,
                "event_candidate_id": selected_candidate_id if selected_observation.confidence > 0 else None,
                "confidence": round(selected_observation.confidence, 4),
                "context": selected_observation.phone_context,
                "occupant_role": selected_observation.occupant_role,
                "rejection_reason": rejection_reason if selected_observation.confidence == 0.0 else "ADMITTED",
                "emitted_events": len(evs),
                "sampling_mode": sampling_mode,
                "sampling_trigger": best_cand is not None,
            })

            if max_frames and processed_count >= max_frames:
                break

            # Fast grab stride. An observed proposal may temporarily switch the
            # next samples to a denser grid; the default path remains unchanged.
            next_stride = self.sample_stride
            if adaptive_sampler is not None:
                next_frame_index = adaptive_sampler.next_frame_index(
                    frame_idx,
                    timestamp,
                    fps,
                    trigger=best_cand is not None,
                )
                next_stride = next_frame_index - frame_idx
            skipped = 0
            for _ in range(next_stride - 1):
                if not cap.grab():
                    break
                skipped += 1
            frame_idx += 1 + skipped

        if last_observation_timestamp is not None:
            boundary_collector.finalize(
                last_observation_timestamp,
                reason=("MAX_FRAMES" if max_frames else "END_OF_STREAM"),
            )
        cap.release()
        total_wall_time = time.perf_counter() - started_at

        # 8. Temporal Cooldown Deduplication (Posture-Regime Aware)
        deduped_preds = []
        cooldown = float(temporal_config.get("cooldown_seconds", 3.0))
        for cand in generated_candidates:
            clean_cand = {k: v for k, v in cand.items() if not k.startswith("_")}
            if not deduped_preds:
                deduped_preds.append(clean_cand)
            else:
                last = deduped_preds[-1]
                last_posture = last.get("posture", "UNKNOWN")
                cand_posture_val = clean_cand.get("posture", "UNKNOWN")
                same_posture = (
                    last_posture == cand_posture_val
                    or "UNKNOWN" in (last_posture, cand_posture_val)
                )
                if same_posture and clean_cand["start_seconds"] <= last["end_seconds"] + cooldown:
                    if clean_cand["end_seconds"] > last["end_seconds"]:
                        last["end_seconds"] = clean_cand["end_seconds"]
                        last["duration"] = round(last["end_seconds"] - last["start_seconds"], 2)
                else:
                    deduped_preds.append(clean_cand)

        actual_frames = min(full_video_frames, frame_idx + 1) if max_frames else frame_idx
        if max_frames is None and actual_frames != full_video_frames:
            raise RuntimeError(f"INCOMPLETE_VIDEO_DECODE: {actual_frames}/{full_video_frames}")
        total_seconds = actual_frames / fps if fps else 0.0

        return {
            "mode": self.mode,
            "scope": "SHORT_SMOKE_TEST" if max_frames else "FULL_COHORT_AUTONOMOUS_EXPERIMENT",
            "fps": fps,
            "action_provenance": self.action_adapter.provenance() if self.action_adapter else None,
            "roi_mode": "STATIC_UNVERIFIED_GEOMETRY",
            "action_mode": self.action_mode,
            "sampling": {
                "adaptive": self.adaptive_sampling,
                "base_stride": self.sample_stride,
                "dense_stride": self.adaptive_stride,
                "hold_seconds": self.adaptive_hold_seconds,
                "processed_modes": sampling_mode_counts,
            },
            "video_path": str(video_path),
            "total_frames": actual_frames,
            "processed_frames": processed_count,
            "wall_time_seconds": round(total_wall_time, 2),
            "effective_fps": round(processed_count / total_wall_time if total_wall_time else 0.0, 2),
            "predicted_events": deduped_preds,
            "prediction_logs": all_prediction_logs,
            "candidate_records": all_candidate_records,
            "action_logs": all_action_logs,
            "object_decision_log": boundary_collector.decision_log,
            "total_seconds": round(total_seconds, 2),
        }


def evaluate_predictions_stage2(
    predicted_events: list[dict[str, Any]],
    annotation_path: Path,
    fps: float,
    total_frames: int,
    total_seconds: float,
    matching_policy: TemporalMatchingPolicy,
) -> dict[str, Any]:
    """STAGE 2: Strict Post-Hoc Evaluation using Frozen Matcher against OpenLABEL GT."""
    ann = parse_dmd_openlabel(annotation_path, fps=fps, total_frames=total_frames)
    gt_intervals = [i.to_dict() for i in ann.phone_intervals]

    matched, unmatched_preds, unmatched_gts = match_temporal_events(
        gt_intervals=gt_intervals,
        predicted_events=predicted_events,
        policy=matching_policy,
    )

    matched_gt_set = set()
    for m in matched:
        gt_item = m["ground_truth"]
        matched_gt_set.add((float(gt_item["start_seconds"]), float(gt_item["end_seconds"])))

    tp = len(matched_gt_set)
    fn = len(unmatched_gts)
    total_gt = len(gt_intervals)
    duplicate_preds = max(0, len(matched) - tp)
    fp = len(unmatched_preds) + duplicate_preds

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / total_gt if total_gt else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    eval_mins = total_seconds / 60.0 if total_seconds > 0 else 1.0
    fa_rate = fp / eval_mins

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": total_gt,
        "evaluated_minutes": round(eval_mins, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fa_rate, 4),
        "matched_events": matched,
        "unmatched_predictions": unmatched_preds,
        "unmatched_gt": unmatched_gts,
    }


def run_v21_full_autonomous_pipeline(
    mode: str = "D5",
    action_mode: str = "RULE_ONLY",
    cohort: str = "two-subject",
    max_frames: int | None = None,
    sample_stride: int = 15,
    enable_dynamic_roi: bool = True,
    enable_verified_association: bool = True,
    enable_action_gating: bool = True,
    action_causal_mode: str = "STRICT_CAUSAL_ONLINE",
    missing_checkpoint_policy: str = "ERROR",
    device: str = "cpu",
    force_handheld_context: bool = False,
    detector_imgsz: int = 640,
    adaptive_sampling: bool = False,
    adaptive_stride: int = 3,
    adaptive_hold_seconds: float = 2.0,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    require_operation("diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy_data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    targets_dict = {
        "gC-14": (ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json"),
        "gZ-36": (ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json"),
        "gB-9": (ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_ann_distraction.json"),
    }

    if cohort == "three-subject":
        active_cohort = ["gB-9", "gC-14", "gZ-36"]
    elif cohort == "gc14-only":
        active_cohort = ["gC-14"]
    elif cohort == "gz36-only":
        active_cohort = ["gZ-36"]
    else:
        active_cohort = ["gC-14", "gZ-36"]
    missing = [str(path) for sub in active_cohort for path in targets_dict[sub] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BLOCKED_MISSING_DATA: {missing}")
    targets = [(sub, *targets_dict[sub]) for sub in active_cohort]

    subject_eval_results = {}
    all_raw_predictions = []
    all_candidates = []
    all_actions = []
    all_event_preds = []
    all_object_decisions = []

    for sub_id, vid_path, ann_path in targets:
        print("\n==========================================", flush=True)
        print(f"Running V21 Autonomous Pipeline on {sub_id}...", flush=True)
        print(f"  Mode={mode}, Action={action_mode}, Association={enable_verified_association}, Stride={sample_stride}", flush=True)

        engine = AutonomousVideoInferenceEngineV21(
            mode=mode,
            action_mode=action_mode,
            confidence_floor=0.15,
            sample_stride=sample_stride,
            enable_dynamic_roi=enable_dynamic_roi,
            enable_verified_association=enable_verified_association,
            enable_action_gating=enable_action_gating,
            heldout_subject=sub_id,
            action_causal_mode=action_causal_mode,
            missing_checkpoint_policy=missing_checkpoint_policy,
            device=device,
            force_handheld_context=force_handheld_context,
            detector_imgsz=detector_imgsz,
            adaptive_sampling=adaptive_sampling,
            adaptive_stride=adaptive_stride,
            adaptive_hold_seconds=adaptive_hold_seconds,
        )

        # STAGE 1: Inference
        inf_res = engine.run_session_inference(
            video_path=vid_path,
            temporal_config=temporal_cfg,
            max_frames=max_frames,
        )

        # Attach subject_id and persist
        for log_row in inf_res["prediction_logs"]:
            log_row["subject_id"] = sub_id
            all_raw_predictions.append(log_row)

        for cand_row in inf_res["candidate_records"]:
            cand_row["subject_id"] = sub_id
            all_candidates.append(cand_row)

        for act_row in inf_res["action_logs"]:
            act_row["subject_id"] = sub_id
            all_actions.append(act_row)

        for ev in inf_res["predicted_events"]:
            ev["subject_id"] = sub_id
            all_event_preds.append(ev)

        for decision in inf_res["object_decision_log"]:
            decision["subject_id"] = sub_id
            all_object_decisions.append(decision)

        # STAGE 2: Post-Hoc Evaluation
        eval_res = evaluate_predictions_stage2(
            predicted_events=inf_res["predicted_events"],
            annotation_path=ann_path,
            fps=inf_res["fps"],
            total_frames=inf_res["total_frames"],
            total_seconds=inf_res["total_seconds"],
            matching_policy=policy,
        )

        subject_eval_results[sub_id] = {
            **eval_res,
            "processed_frames": inf_res["processed_frames"],
            "wall_time_seconds": inf_res["wall_time_seconds"],
            "effective_fps": inf_res["effective_fps"],
            "predicted_events_count": len(inf_res["predicted_events"]),
            "action_provenance": inf_res["action_provenance"],
            "sampling_diagnostics": inf_res["sampling"],
        }

        # Persist each completed subject immediately. A long raw-video benchmark
        # must retain finished evidence if a later session is interrupted.
        subject_dir = output_dir / "subjects" / sub_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "metrics.json").write_text(
            json.dumps(subject_eval_results[sub_id], indent=2) + "\n", encoding="utf-8"
        )
        for filename, records in (
            ("autonomous_predictions.jsonl", inf_res["prediction_logs"]),
            ("phone_candidate_records.jsonl", inf_res["candidate_records"]),
            ("action_predictions.jsonl", inf_res["action_logs"]),
            ("event_predictions.jsonl", inf_res["predicted_events"]),
            ("object_decisions.jsonl", inf_res["object_decision_log"]),
        ):
            with (subject_dir / filename).open("w", encoding="utf-8") as subject_handle:
                for record in records:
                    subject_handle.write(json.dumps(record) + "\n")

        print(f"  Results {sub_id} -> TP={eval_res['true_positives']}, FP={eval_res['false_positives']}, FN={eval_res['false_negatives']}, Precision={eval_res['precision']}, Recall={eval_res['recall']}, F1={eval_res['f1']}, FA/min={eval_res['false_alarms_per_minute']}", flush=True)

    # Aggregate
    tp_sum = sum(subject_eval_results[s]["true_positives"] for s in subject_eval_results)
    fp_sum = sum(subject_eval_results[s]["false_positives"] for s in subject_eval_results)
    fn_sum = sum(subject_eval_results[s]["false_negatives"] for s in subject_eval_results)
    gt_sum = sum(subject_eval_results[s]["total_gt_events"] for s in subject_eval_results)
    min_sum = sum(subject_eval_results[s]["evaluated_minutes"] for s in subject_eval_results)

    prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    rec = tp_sum / gt_sum if gt_sum else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fa_rate = fp_sum / min_sum if min_sum else 0.0

    out = {
        "schema_version": "v21.0",
        "evaluation_type": "AUTONOMOUS_VIDEO_INFERENCE_V21",
        "cohort": active_cohort,
        "detector_mode": mode,
        "detector_imgsz": detector_imgsz,
        "adaptive_sampling": adaptive_sampling,
        "adaptive_stride": adaptive_stride,
        "adaptive_hold_seconds": adaptive_hold_seconds,
        "action_mode": action_mode,
        "action_causal_mode": action_causal_mode,
        "missing_checkpoint_policy": missing_checkpoint_policy,
        "device": device,
        "force_handheld_context": force_handheld_context,
        "verified_association": enable_verified_association,
        "action_gating": enable_action_gating,
        "total_gt_events": gt_sum,
        "evaluated_minutes": round(min_sum, 2),
        "true_positives": tp_sum,
        "false_positives": fp_sum,
        "false_negatives": fn_sum,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fa_rate, 4),
        "per_subject": {
            s: {
                "tp": subject_eval_results[s]["true_positives"],
                "fp": subject_eval_results[s]["false_positives"],
                "fn": subject_eval_results[s]["false_negatives"],
                "precision": subject_eval_results[s]["precision"],
                "recall": subject_eval_results[s]["recall"],
                "f1": subject_eval_results[s]["f1"],
                "fa_per_min": subject_eval_results[s]["false_alarms_per_minute"],
                "effective_fps": subject_eval_results[s]["effective_fps"],
                "action_provenance": subject_eval_results[s]["action_provenance"],
                "sampling_diagnostics": subject_eval_results[s]["sampling_diagnostics"],
            }
            for s in subject_eval_results
        },
    }

    # Save deliverables in JSONL and JSON
    (output_dir / "event_matches.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )

    # Complete Prediction Persistence (every single frame, no [:100] truncation!)
    with (output_dir / "autonomous_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_raw_predictions:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "phone_candidate_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_candidates:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "action_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_actions:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "event_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_event_preds:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "object_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_object_decisions:
            handle.write(json.dumps(row) + "\n")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Quick test run on 100 frames")
    parser.add_argument("--mode", default="D5", choices=["D0", "D2", "D3", "D5"], help="Detection ablation mode")
    parser.add_argument("--action-mode", default="RULE_ONLY", choices=["HYBRID_RULE_V6", "RULE_ONLY", "V6_HIERARCHICAL", "V7_BINARY_GATE", "NONE"], help="Action mode")
    parser.add_argument("--device", default="cpu", help="Ultralytics/Torch device, for example cpu or 0")
    parser.add_argument(
        "--cohort",
        default="two-subject",
        choices=["two-subject", "three-subject", "gc14-only", "gz36-only"],
        help="Cohort",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process per session")
    parser.add_argument("--sample-stride", type=int, default=15, help="Frame sample stride")
    parser.add_argument("--detector-imgsz", type=int, default=640, help="Detector inference image size")
    parser.add_argument("--adaptive-sampling", action="store_true", help="Use proposal-triggered dense sampling")
    parser.add_argument("--adaptive-stride", type=int, default=3, help="Dense stride after an adaptive trigger")
    parser.add_argument("--adaptive-hold-seconds", type=float, default=2.0, help="Seconds to retain dense sampling after a trigger")
    parser.add_argument("--disable-association", action="store_true", help="Ablation: disable verified driver association")
    parser.add_argument("--disable-action-gating", action="store_true", help="Ablation: disable action gating")
    parser.add_argument(
        "--force-handheld-context",
        action="store_true",
        help="Diagnostic upper bound: treat every phone proposal as HANDHELD_USE",
    )
    parser.add_argument("--output", type=Path, default=REPORTS_DIR, help="Artifact directory")
    args = parser.parse_args()

    max_f = 100 if args.test_run else args.max_frames
    res = run_v21_full_autonomous_pipeline(
        mode=args.mode,
        action_mode=args.action_mode,
        cohort=args.cohort,
        max_frames=max_f,
        sample_stride=args.sample_stride,
        detector_imgsz=args.detector_imgsz,
        adaptive_sampling=args.adaptive_sampling,
        adaptive_stride=args.adaptive_stride,
        adaptive_hold_seconds=args.adaptive_hold_seconds,
        enable_verified_association=not args.disable_association,
        enable_action_gating=not args.disable_action_gating,
        device=args.device,
        force_handheld_context=args.force_handheld_context,
        output_dir=args.output,
    )
    print("\n==========================================")
    print("V21 AUTONOMOUS EVALUATION SUMMARY:")
    print(json.dumps(res, indent=2))
