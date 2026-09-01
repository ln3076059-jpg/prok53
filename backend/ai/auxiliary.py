from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.ai.detector import NormalizedDetection


@dataclass(frozen=True)
class VehicleRegion:
    context_id: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None


@dataclass(frozen=True)
class PoseEvidence:
    confidence: float
    face_points: tuple[tuple[float, float], ...]
    hand_points: tuple[tuple[float, float], ...]
    xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SeatbeltEvidence:
    fastened: float
    unfastened: float
    uncertain_or_occluded: float
    source: str


def _point_box_proximity(
    points: tuple[tuple[float, float], ...], box: tuple[float, float, float, float]
) -> float:
    if not points:
        return 0.0
    x1, y1, x2, y2 = box
    diagonal = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    distance = min(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in points)
    return max(0.0, min(1.0, 1.0 - distance / (2.5 * diagonal)))


def phone_context(
    detection: NormalizedDetection,
    pose: PoseEvidence | None,
    hand_threshold: float = 0.55,
    face_threshold: float = 0.55,
) -> tuple[str, float, float]:
    if not pose or (not pose.hand_points and not pose.face_points):
        return "UNKNOWN", 0.0, 0.0
    hand = _point_box_proximity(pose.hand_points, detection.xyxy)
    face = _point_box_proximity(pose.face_points, detection.xyxy)
    if hand >= hand_threshold or face >= face_threshold:
        return "HANDHELD", hand, face
    return "MOUNTED_OR_STATIC", hand, face


class VehicleDetector:
    COCO_VEHICLE_CLASSES = {2, 3, 5, 7}

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

    def predict(self, frame: np.ndarray) -> list[VehicleRegion]:
        self._load()
        result = self._model.track(
            frame,
            persist=True,
            classes=sorted(self.COCO_VEHICLE_CLASSES),
            conf=self.confidence,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        regions = []
        for index, (xyxy, confidence, track_id) in enumerate(
            zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), ids, strict=True)
        ):
            normalized_id = int(track_id) if track_id is not None else None
            context_id = f"vehicle:{normalized_id}" if normalized_id is not None else f"vehicle:frame:{index}"
            regions.append(
                VehicleRegion(
                    context_id,
                    float(confidence),
                    tuple(float(value) for value in xyxy),
                    normalized_id,
                )
            )
        return regions


class PoseEstimator:
    # COCO pose indices: nose/eyes/ears, then left/right wrist.
    FACE_INDICES = (0, 1, 2, 3, 4)
    HAND_INDICES = (9, 10)

    def __init__(self, weights: Path | None, confidence: float = 0.25):
        self.weights = weights
        self.confidence = confidence
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
        result = self._model.predict(frame, conf=self.confidence, verbose=False)[0]
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
        for points, confidences, pose_box in zip(
            xy_rows, confidence_rows, pose_boxes, strict=True
        ):
            def selected(
                indices: tuple[int, ...],
                row_points=points,
                row_confidences=confidences,
            ) -> tuple[tuple[float, float], ...]:
                return tuple(
                    (float(row_points[index][0]), float(row_points[index][1]))
                    for index in indices
                    if index < len(row_points)
                    and float(row_confidences[index]) >= self.confidence
                )

            valid = [float(value) for value in confidences if float(value) >= self.confidence]
            evidence.append(
                PoseEvidence(
                    confidence=sum(valid) / len(valid) if valid else 0.0,
                    face_points=selected(self.FACE_INDICES),
                    hand_points=selected(self.HAND_INDICES),
                    xyxy=(tuple(float(value) for value in pose_box) if pose_box else None),
                )
            )
        return evidence


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
        result = self._model.predict(crop, verbose=False)[0]
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


def crop_box(frame: np.ndarray, xyxy: tuple[float, float, float, float], padding: float = 0.08) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    return frame[top:bottom, left:right]
