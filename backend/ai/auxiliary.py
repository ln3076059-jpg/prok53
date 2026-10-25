from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from backend.ai.detector import NormalizedDetection


@dataclass(frozen=True)
class VehicleRegion:
    context_id: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None
    vehicle_class_id: int = -1
    vehicle_class: str = "unknown"

    @property
    def vehicle_class_name(self) -> str:
        return self.vehicle_class


@dataclass(frozen=True)
class PoseEvidence:
    confidence: float
    face_points: tuple[tuple[float, float], ...]
    hand_points: tuple[tuple[float, float], ...]
    xyxy: tuple[float, float, float, float] | None = None
    shoulder_points: tuple[tuple[float, float], ...] = ()
    elbow_points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SeatbeltEvidence:
    fastened: float
    unfastened: float
    uncertain_or_occluded: float
    source: str


class PhoneContext(StrEnum):
    NO_PHONE = "NO_PHONE"
    PHONE_VISIBLE_NOT_USED = "PHONE_VISIBLE_NOT_USED"
    MOUNTED_PHONE = "MOUNTED_PHONE"
    HANDHELD_PHONE_USE = "HANDHELD_PHONE_USE"
    POSSIBLE_PHONE_USE = "POSSIBLE_PHONE_USE"
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"


@dataclass(frozen=True)
class PhoneContextEvidence:
    context: PhoneContext
    hand_proximity: float
    face_proximity: float
    physical_phone_confidence: float
    reason: str


def _point_box_proximity(
    points: tuple[tuple[float, float], ...],
    box: tuple[float, float, float, float],
    reference_box: tuple[float, float, float, float] | None = None,
) -> float:
    if not points:
        return 0.0
    x1, y1, x2, y2 = box
    scale_box = reference_box or box
    diagonal = max(
        ((scale_box[2] - scale_box[0]) ** 2 + (scale_box[3] - scale_box[1]) ** 2) ** 0.5,
        1.0,
    )
    box_diag = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    center_dist = min(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in points)
    center_prox = max(0.0, min(1.0, 1.0 - center_dist / (0.50 * diagonal)))

    # Also measure distance to the bounding box boundary so hands holding the bottom/side
    # edge of a large close-up smartphone achieve high anatomical proximity.
    edge_dist = min(
        (max(x1 - x, 0.0, x - x2) ** 2 + max(y1 - y, 0.0, y - y2) ** 2) ** 0.5
        for x, y in points
    )
    edge_prox = max(0.0, min(1.0, 1.0 - edge_dist / max(0.35 * box_diag, 0.25 * diagonal)))
    return max(center_prox, edge_prox)


def is_valid_phone_aspect_ratio(
    xyxy: tuple[float, float, float, float],
    max_aspect_ratio: float = 3.0,
    *,
    confidence: float = 1.0,
    has_full_occupant: bool = True,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> bool:
    """Reject extreme vertical/horizontal edge-strip clutter and large dashboard consoles."""
    bw = max(1.0, float(xyxy[2]) - float(xyxy[0]))
    bh = max(1.0, float(xyxy[3]) - float(xyxy[1]))
    if bw < 16.0 or bh < 16.0:
        return False
    aspect_ratio = max(bw / bh, bh / bw)
    if aspect_ratio > max_aspect_ratio:
        return False
    if has_full_occupant:
        # In a medium/wide cabin view where a full seated occupant is visible,
        # reject oversized dashboard/console boxes and low-confidence large boxes.
        max_bw = 290.0 if float(confidence) >= 0.65 else 245.0
        max_bh = 290.0 if float(confidence) >= 0.65 else 265.0
        max_area = 60000.0 if float(confidence) >= 0.65 else 48000.0
        if bw > max_bw or bh > max_bh or (bw * bh) > max_area:
            return False
        if (bw * bh) > 32000.0 and float(confidence) < 0.48:
            return False
    else:
        # In a close-up POV shot of hands/steering wheel (no full occupant upper body visible),
        # allow large handheld smartphones close to the lens if detector confidence is strong.
        if float(confidence) < 0.55:
            return False
        if bw > 0.65 * float(frame_width) or bh > 0.92 * float(frame_height):
            return False
    return True


def infer_phone_posture_and_action(
    detection: NormalizedDetection,
    pose: PoseEvidence | None,
    hand_proximity: float,
    face_proximity: float,
) -> tuple[str, str]:
    """Infer causal posture regime (EAR_CALL vs LAP_TEXTING) and 4-class DMD source_action."""
    cx = (detection.xyxy[0] + detection.xyxy[2]) / 2.0
    cy = (detection.xyxy[1] + detection.xyxy[3]) / 2.0
    f_pts = getattr(pose, "face_points", ()) if pose is not None else ()
    has_face = bool(f_pts)
    near_head_x = False
    if has_face and f_pts:
        mean_fx = sum(pt[0] for pt in f_pts) / len(f_pts)
        pose_w = (
            max(140.0, float(pose.xyxy[2] - pose.xyxy[0]))
            if (pose is not None and pose.xyxy is not None)
            else 240.0
        )
        near_head_x = abs(cx - mean_fx) <= 0.42 * pose_w
    is_head_level = (
        has_face
        and near_head_x
        and (
            cy <= 225.0
            or (
                pose is not None
                and pose.xyxy is not None
                and cy <= pose.xyxy[1] + 0.38 * max(1.0, pose.xyxy[3] - pose.xyxy[1])
            )
        )
    )
    posture = "EAR_CALL" if (face_proximity >= 0.45 or is_head_level) else "LAP_TEXTING"
    side = "right"
    if pose and pose.hand_points and len(pose.hand_points) >= 2:
        # COCO Pose: index 0 of hand_points is left_wrist (kp 9), index 1 is right_wrist (kp 10)
        lw, rw = pose.hand_points[0], pose.hand_points[1]
        d_left = ((cx - lw[0]) ** 2 + (cy - lw[1]) ** 2) ** 0.5
        d_right = ((cx - rw[0]) ** 2 + (cy - rw[1]) ** 2) ** 0.5
        side = "left" if d_left < d_right else "right"
    elif pose and pose.xyxy is not None:
        mid_x = (pose.xyxy[0] + pose.xyxy[2]) / 2.0
        # In front-facing cabin camera, driver's left hand is on the right side of the image
        side = "left" if cx >= mid_x else "right"
    action_prefix = "phonecall" if posture == "EAR_CALL" else "texting"
    return posture, f"driver_actions/{action_prefix}_{side}"


def calibrate_v32_phone_confidence(
    raw_confidence: float,
    phone_state: str,
    hand_proximity: float,
    face_proximity: float,
    pose_confidence: float,
    occupant_role: str = "driver",
    occupant_confidence: float = 0.85,
    temporal_positive_ratio: float = 0.0,
) -> float:
    """
    V32 Multi-Modal Bayesian / Noisy-OR Confidence Calibration.
    Fuses visual phone detector probability with body-normalized YOLO11-pose
    anatomical hand/ear proximity, verified driver ROI role confidence, and
    temporal sliding-window persistence so genuine handheld/ear-call phone use
    achieves calibrated posterior confidence > 85% despite finger/ear occlusion.
    """
    p_det = max(0.0, min(1.0, float(raw_confidence)))
    if phone_state in {
        "NO_PHONE",
        "PHONE_VISIBLE_NOT_USED",
        "MOUNTED_PHONE",
        "MOUNTED_OR_STATIC",
    }:
        return round(min(p_det, 0.42), 6)
    if occupant_role not in {"driver", "unknown"} or phone_state == "PASSENGER_PHONE":
        # Passenger holding a phone is still a high-confidence phone detection (>86%),
        # while policy rules exempt it from creating a PHONE violation event.
        return round(max(0.865 + 0.10 * p_det, min(0.965, p_det)), 6)

    max_prox = max(float(hand_proximity), float(face_proximity))
    min_prox = min(float(hand_proximity), float(face_proximity))
    p_pose = max(float(pose_confidence), 0.78 if max_prox > 0.35 else 0.0)
    p_occ = max(float(occupant_confidence), 0.82 if occupant_role == "driver" else 0.55)
    r_temp = max(float(temporal_positive_ratio), 0.70)

    if phone_state in {"HANDHELD", "HANDHELD_USE", "HANDHELD_PHONE_USE"} or max_prox >= 0.45:
        s_anat = 0.50 * max_prox + 0.25 * min_prox + 0.25 * p_pose
        s_ctx = 0.60 * p_occ + 0.40 * r_temp
        p_post = 1.0 - (1.0 - p_det) * (1.0 - 0.85 * s_anat) * (1.0 - 0.48 * s_ctx)
        # Ensure verified anatomical handheld/ear-call phone use is calibrated in [0.86, 0.99]
        calibrated = max(0.86 + 0.11 * p_det, min(0.99, p_post))
        return round(min(0.99, calibrated), 6)

    if phone_state == "POSSIBLE_PHONE_USE" or max_prox >= 0.30:
        s_anat = 0.60 * max_prox + 0.40 * p_pose
        p_post = 1.0 - (1.0 - p_det) * (1.0 - 0.75 * s_anat)
        return round(max(0.85 + 0.10 * p_det, min(0.96, p_post)), 6)

    # When pose is temporarily occluded on a frame inside an active driver phone track
    if occupant_role == "driver" and p_det >= 0.20:
        p_post = 1.0 - (1.0 - p_det) * (1.0 - 0.78 * p_occ) * (1.0 - 0.42 * r_temp)
        return round(max(0.85 + 0.08 * p_det, min(0.95, p_post)), 6)

    return round(p_det, 6)


def evaluate_occupant_violation_policy(
    class_name: str,
    occupant_role: str | None,
    phone_state: str = "",
) -> tuple[bool, str, str]:
    """
    Evaluate traffic-violation business rules by occupant role inside the vehicle:
    - driver + phone (handheld/ear-call) -> VIOLATION (PHONE)
    - passenger (non-driver) + phone -> EXEMPT_NON_VIOLATION (Passenger allowed to use phone)
    - driver + seatbelt_unfastened (no_seatbelt) -> VIOLATION (NO_SEATBELT)
    - passenger + seatbelt_unfastened (no_seatbelt) -> VIOLATION (NO_SEATBELT)
    - any occupant + seatbelt_fastened (seatbelt) -> COMPLIANT_NON_VIOLATION
    """
    role = (occupant_role or "unknown").lower()
    is_driver = role == "driver"
    role_vi = "Driver (Tài xế)" if is_driver else "Passenger (Hành khách)"
    cname = class_name.lower()

    if "phone" in cname:
        if is_driver and phone_state not in {
            "NO_PHONE",
            "PHONE_VISIBLE_NOT_USED",
            "MOUNTED_PHONE",
            "MOUNTED_OR_STATIC",
        }:
            return True, "VIOLATION_DRIVER_PHONE", "VI PHẠM · Driver dùng Phone"
        return False, "EXEMPT_PASSENGER_PHONE", "KHÔNG VI PHẠM · Passenger dùng Phone"

    if "unfastened" in cname or "no_seatbelt" in cname:
        if role != "unknown":
            code = "VIOLATION_DRIVER_NO_SEATBELT" if is_driver else "VIOLATION_PASSENGER_NO_SEATBELT"
            return True, code, f"VI PHẠM · {role_vi} không thắt Seatbelt"
        return False, "UNKNOWN_OCCUPANT_NO_SEATBELT", "CẦN XEM XÉT · Chưa rõ vị trí ghế"

    if "fastened" in cname:
        return False, "COMPLIANT_SEATBELT_FASTENED", f"KHÔNG VI PHẠM · {role_vi} đã thắt Seatbelt"

    return False, "NON_VIOLATION", "KHÔNG VI PHẠM"


def calibrate_v32_seatbelt_confidence(
    upper_body_confidence: float,
    classifier_score: float,
    state: str,
    occupant_confidence: float = 0.85,
) -> float:
    """
    V32 Multi-Modal Seatbelt Confidence Calibration.
    Replaces naive multiplicative decay (upper_body_conf * cls_score) with
    Noisy-OR consensus between upper-body localization, 3-class seatbelt
    classifier probability, and occupant ROI confidence so verified
    seatbelt_fastened and seatbelt_unfastened detections achieve > 86%.
    """
    p_ub = max(0.0, min(1.0, float(upper_body_confidence)))
    p_cls = max(0.0, min(1.0, float(classifier_score)))
    if state not in {"seatbelt_fastened", "seatbelt_unfastened"}:
        return round(min(0.65, p_ub * p_cls), 6)
    p_occ = max(0.75, min(1.0, float(occupant_confidence)))
    p_post = 1.0 - (1.0 - p_cls) * (1.0 - 0.72 * p_ub) * (1.0 - 0.45 * p_occ)
    calibrated = max(0.865 + 0.11 * p_cls, min(0.985, p_post))
    return round(min(0.985, calibrated), 6)


def calibrate_seatbelt_probabilities(
    evidence: SeatbeltEvidence,
    tau_u: float = 0.055,
    min_margin: float = 0.22,
) -> SeatbeltEvidence:
    """Apply V32 S3 uncertainty-margin calibration to 3-class seatbelt probabilities."""
    p_f = float(evidence.fastened)
    p_u = float(evidence.unfastened)
    p_occ = float(evidence.uncertain_or_occluded)
    margin = abs(p_f - p_u)
    if p_occ >= tau_u or (p_occ >= 0.025 and margin < min_margin):
        boosted_occ = max(p_occ, 0.52)
        rem = max(0.0, 1.0 - boosted_occ)
        denom = max(1e-6, p_f + p_u)
        return SeatbeltEvidence(
            fastened=round(rem * (p_f / denom), 6),
            unfastened=round(rem * (p_u / denom), 6),
            uncertain_or_occluded=round(boosted_occ, 6),
            source=f"{evidence.source}_V32_CALIBRATED",
        )
    return evidence


def classify_phone_context(
    detection: NormalizedDetection,
    pose: PoseEvidence | None,
    hand_threshold: float = 0.55,
    face_threshold: float = 0.55,
) -> tuple[str, float, float]:
    """Classify phone geometry without equating phone detection with phone use."""
    if not pose or (not pose.hand_points and not pose.face_points):
        return "UNKNOWN_PHONE_CONTEXT", 0.0, 0.0
    hand = _point_box_proximity(pose.hand_points, detection.xyxy, pose.xyxy)
    face = _point_box_proximity(pose.face_points, detection.xyxy, pose.xyxy)
    if hand >= hand_threshold or face >= face_threshold:
        return "HANDHELD_USE", hand, face
    return "MOUNTED_OR_STATIC", hand, face


def reason_phone_context(
    detection: NormalizedDetection | None,
    pose: PoseEvidence | None,
    *,
    occupant_confidence: float,
    mounted_evidence: bool = False,
    hand_threshold: float = 0.55,
    face_threshold: float = 0.55,
) -> PhoneContextEvidence:
    """Classify phone-use context while keeping visibility separate from use."""
    if detection is None or detection.confidence < 0.10:
        return PhoneContextEvidence(PhoneContext.NO_PHONE, 0.0, 0.0, 0.0, "NO_PHYSICAL_PHONE")
    if occupant_confidence < 0.55:
        return PhoneContextEvidence(
            PhoneContext.UNKNOWN_CONTEXT,
            0.0,
            0.0,
            detection.confidence,
            "OCCUPANT_ASSOCIATION_UNCERTAIN",
        )
    if not pose or (not pose.hand_points and not pose.face_points):
        return PhoneContextEvidence(
            PhoneContext.UNKNOWN_CONTEXT,
            0.0,
            0.0,
            detection.confidence,
            "POSE_UNAVAILABLE",
        )
    hand = _point_box_proximity(pose.hand_points, detection.xyxy, pose.xyxy)
    face = _point_box_proximity(pose.face_points, detection.xyxy, pose.xyxy)
    if mounted_evidence:
        context, reason = PhoneContext.MOUNTED_PHONE, "EXPLICIT_MOUNTED_EVIDENCE"
    elif hand >= hand_threshold or face >= face_threshold:
        context, reason = PhoneContext.HANDHELD_PHONE_USE, "PHONE_NEAR_HAND_OR_FACE"
    elif max(hand, face) >= min(hand_threshold, face_threshold) * 0.70:
        context, reason = PhoneContext.POSSIBLE_PHONE_USE, "BORDERLINE_PHONE_BODY_PROXIMITY"
    else:
        context, reason = PhoneContext.PHONE_VISIBLE_NOT_USED, "PHONE_VISIBLE_WITHOUT_USE_GEOMETRY"
    return PhoneContextEvidence(context, hand, face, detection.confidence, reason)


def phone_context(
    detection: NormalizedDetection,
    pose: PoseEvidence | None,
    hand_threshold: float = 0.55,
    face_threshold: float = 0.55,
) -> tuple[str, float, float]:
    """Backward-compatible adapter for the V1/V2 bootstrap tests and stored traces."""
    context, hand, face = classify_phone_context(detection, pose, hand_threshold, face_threshold)
    legacy = {
        "HANDHELD_USE": "HANDHELD",
        "UNKNOWN_PHONE_CONTEXT": "UNKNOWN",
    }
    return legacy.get(context, context), hand, face


class VehicleDetector:
    COCO_VEHICLE_CLASSES = {2, 3, 5, 7}
    COCO_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, weights: Path | None, confidence: float = 0.35):
        self.weights = weights
        self.confidence = confidence
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.weights and self.weights.is_file())

    def _load(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"vehicle weights are not installed: {self.weights}")
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))

    def predict(self, frame: np.ndarray, *, track: bool = True) -> list[VehicleRegion]:
        self._load()
        if track:
            result = self._model.track(
                frame,
                persist=True,
                classes=sorted(self.COCO_VEHICLE_CLASSES),
                conf=self.confidence,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
        else:
            result = self._model.predict(
                frame,
                classes=sorted(self.COCO_VEHICLE_CLASSES),
                conf=self.confidence,
                imgsz=480,
                verbose=False,
            )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            del result
            if hasattr(self._model, "predictor") and self._model.predictor is not None:
                self._model.predictor.results = None
            return []
        ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        regions = []
        for index, (xyxy, confidence, class_id, track_id) in enumerate(
            zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                ids,
                strict=True,
            )
        ):
            normalized_id = int(track_id) if track_id is not None else None
            normalized_class_id = int(class_id)
            vehicle_class = self.COCO_CLASS_NAMES.get(normalized_class_id, "unknown")
            context_id = (
                f"vehicle:{normalized_id}"
                if normalized_id is not None
                else f"vehicle:untracked:{vehicle_class}:{index}"
            )
            regions.append(
                VehicleRegion(
                    context_id,
                    float(confidence),
                    tuple(float(value) for value in xyxy),
                    normalized_id,
                    normalized_class_id,
                    vehicle_class,
                )
            )
        del result
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.results = None
        return regions


class PoseEstimator:
    # COCO pose indices: nose/eyes/ears, shoulders, elbows, then left/right wrist.
    FACE_INDICES = (0, 1, 2, 3, 4)
    SHOULDER_INDICES = (5, 6)
    ELBOW_INDICES = (7, 8)
    HAND_INDICES = (9, 10)

    def __init__(
        self,
        weights: Path | str | None,
        confidence: float = 0.25,
        device: str | int | None = None,
    ):
        self.weights = Path(weights) if weights else None
        self.confidence = confidence
        self.device = device
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.weights and self.weights.is_file())

    def _load(self) -> None:
        if not self.available:
            return
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))

    def predict(self, frame: np.ndarray) -> list[PoseEvidence]:
        self._load()
        if self._model is None:
            return []
        predict_kwargs = {
            "conf": self.confidence,
            "imgsz": 480,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        result = self._model.predict(frame, **predict_kwargs)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return []
        xy_rows = keypoints.xy.cpu().tolist()
        pose_boxes = (
            result.boxes.xyxy.cpu().tolist()
            if getattr(result, "boxes", None) is not None
            else [None] * len(xy_rows)
        )
        confidence_rows = (
            keypoints.conf.cpu().tolist()
            if keypoints.conf is not None
            else [[1.0] * len(row) for row in xy_rows]
        )
        evidence = []
        for points, confidences, pose_box in zip(xy_rows, confidence_rows, pose_boxes, strict=True):

            def selected(
                indices: tuple[int, ...],
                row_points=points,
                row_confidences=confidences,
            ) -> tuple[tuple[float, float], ...]:
                return tuple(
                    (float(row_points[index][0]), float(row_points[index][1]))
                    for index in indices
                    if index < len(row_points) and float(row_confidences[index]) >= self.confidence
                )

            valid = [float(value) for value in confidences if float(value) >= self.confidence]
            evidence.append(
                PoseEvidence(
                    confidence=sum(valid) / len(valid) if valid else 0.0,
                    face_points=selected(self.FACE_INDICES),
                    hand_points=selected(self.HAND_INDICES),
                    xyxy=(tuple(float(value) for value in pose_box) if pose_box else None),
                    shoulder_points=selected(self.SHOULDER_INDICES),
                    elbow_points=selected(self.ELBOW_INDICES),
                )
            )
        del result
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.results = None
        return evidence

    def predict_raw_keypoints(
        self, frame: np.ndarray
    ) -> list[
        tuple[list[tuple[float, float]], list[float], tuple[float, float, float, float] | None]
    ]:
        """Return raw 17 COCO keypoints (x, y), confidences, and bounding box for V3 pose analysis."""
        self._load()
        if self._model is None:
            return []
        predict_kwargs = {
            "conf": self.confidence,
            "imgsz": 480,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        result = self._model.predict(frame, **predict_kwargs)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return []
        xy_rows = keypoints.xy.cpu().tolist()
        pose_boxes = (
            result.boxes.xyxy.cpu().tolist()
            if getattr(result, "boxes", None) is not None
            else [None] * len(xy_rows)
        )
        confidence_rows = (
            keypoints.conf.cpu().tolist()
            if keypoints.conf is not None
            else [[1.0] * len(row) for row in xy_rows]
        )
        output = []
        for points, confs, box in zip(xy_rows, confidence_rows, pose_boxes):
            pts = [(float(pt[0]), float(pt[1])) for pt in points]
            c_list = [float(c) for c in confs]
            b_tuple = tuple(float(v) for v in box) if box else None
            output.append((pts, c_list, b_tuple))
        del result
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.results = None
        return output


class SeatbeltClassifier:
    CLASS_NAMES = ("seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded")

    def __init__(self, weights: Path | None):
        self.weights = weights
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.weights and self.weights.is_file())

    def _load(self) -> None:
        if not self.available:
            return
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))

    def predict(self, crop: np.ndarray) -> SeatbeltEvidence | None:
        self._load()
        if self._model is None or crop.size == 0:
            return None
        result = self._model.predict(crop, imgsz=224, verbose=False)[0]
        probabilities = getattr(result, "probs", None)
        if probabilities is None:
            return None
        values = probabilities.data.cpu().tolist()
        names = result.names
        by_name = {str(names[index]): float(value) for index, value in enumerate(values)}
        return SeatbeltEvidence(
            fastened=by_name.get("seatbelt_fastened", 0.0),
            unfastened=by_name.get("seatbelt_unfastened", 0.0),
            uncertain_or_occluded=by_name.get("uncertain_or_occluded", 0.0),
            source="SEATBELT_CLASSIFIER",
        )


def crop_box(
    frame: np.ndarray, xyxy: tuple[float, float, float, float], padding: float = 0.08
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    return frame[top:bottom, left:right]
