from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from backend.ai.detector import NormalizedDetection


@dataclass(frozen=True)
class OccupantRegion:
    role: str
    normalized_xyxy: list[float]
    enabled: bool = True
    min_overlap: float = 0.5


@dataclass(frozen=True)
class OccupantAssignment:
    role: str
    confidence: float
    method: str
    occupant_track_id: int | None = None

    @property
    def known(self) -> bool:
        return self.role != "unknown"


def normalized_roi_to_pixels(
    roi: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    return x1 * width, y1 * height, x2 * width, y2 * height


def intersection_over_box(
    box: tuple[float, float, float, float], roi: tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    intersection = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return 0.0 if area == 0 else intersection / area


def associate_driver(
    detection: NormalizedDetection,
    frame_width: int,
    frame_height: int,
    roi: list[float],
    min_overlap: float = 0.5,
) -> bool:
    return (
        intersection_over_box(
            detection.xyxy, normalized_roi_to_pixels(roi, frame_width, frame_height)
        )
        >= min_overlap
    )


def associate_occupant(
    detection: NormalizedDetection,
    frame_width: int,
    frame_height: int,
    regions: list[dict],
) -> str | None:
    """Return the best calibrated occupant region for a detection.

    Regions intentionally describe image geometry, not identity. Camera-specific calibration
    remains mandatory because left/right image position can flip with camera placement.
    """
    best_role: str | None = None
    best_overlap = 0.0
    for item in regions:
        region = OccupantRegion(**item)
        if not region.enabled:
            continue
        overlap = intersection_over_box(
            detection.xyxy,
            normalized_roi_to_pixels(region.normalized_xyxy, frame_width, frame_height),
        )
        if overlap >= region.min_overlap and overlap > best_overlap:
            best_role = region.role
            best_overlap = overlap
    return best_role


class OccupantAssociator:
    """Associate upper bodies and objects with seats using explicit confidence.

    Camera-calibrated regions remain supported, but geometry and handedness are the primary
    non-model inputs. Low-confidence assignments remain `unknown`; they never default to driver.
    """

    VALID_ROLES = {
        "driver",
        "front_passenger",
        "rear_left",
        "rear_center",
        "rear_right",
        "unknown",
    }

    def __init__(
        self,
        driver_side: str = "left",
        minimum_confidence: float = 0.55,
        rear_seats_enabled: bool = True,
        calibrated_regions: list[dict] | None = None,
    ):
        if driver_side not in {"left", "right"}:
            raise ValueError("driver_side must be left or right in image coordinates")
        self.driver_side = driver_side
        self.minimum_confidence = float(minimum_confidence)
        self.rear_seats_enabled = bool(rear_seats_enabled)
        self.calibrated_regions = calibrated_regions or []

    def assign_upper_body(
        self,
        detection: NormalizedDetection,
        frame_width: int,
        frame_height: int,
    ) -> OccupantAssignment:
        calibrated = self._calibrated(detection, frame_width, frame_height)
        if calibrated.confidence >= self.minimum_confidence:
            return calibrated

        x1, y1, x2, y2 = detection.xyxy
        center_x = ((x1 + x2) / 2) / max(frame_width, 1)
        center_y = ((y1 + y2) / 2) / max(frame_height, 1)
        if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
            return OccupantAssignment("unknown", 0.0, "OUT_OF_CABIN", detection.track_id)

        if self.rear_seats_enabled and center_y >= 0.68:
            if center_x < 0.34:
                role, anchor = "rear_left", (0.18, 0.78)
            elif center_x > 0.66:
                role, anchor = "rear_right", (0.82, 0.78)
            else:
                role, anchor = "rear_center", (0.50, 0.78)
        else:
            driver_on_left = self.driver_side == "left"
            if abs(center_x - 0.5) < 0.08:
                return OccupantAssignment(
                    "unknown", 0.45, "CABIN_GEOMETRY_AMBIGUOUS", detection.track_id
                )
            is_left = center_x < 0.5
            role = "driver" if is_left == driver_on_left else "front_passenger"
            anchor = ((0.25 if is_left else 0.75), 0.48)

        distance = hypot(center_x - anchor[0], center_y - anchor[1])
        confidence = max(0.0, min(0.85, 0.85 - distance))
        if confidence < self.minimum_confidence:
            return OccupantAssignment(
                "unknown", confidence, "CABIN_GEOMETRY_LOW_CONFIDENCE", detection.track_id
            )
        return OccupantAssignment(role, confidence, "CABIN_GEOMETRY", detection.track_id)

    def assign_object(
        self,
        detection: NormalizedDetection,
        occupants: list[tuple[NormalizedDetection, OccupantAssignment]],
        frame_width: int,
        frame_height: int,
    ) -> OccupantAssignment:
        if not occupants:
            calibrated = self._calibrated(detection, frame_width, frame_height)
            if calibrated.confidence >= self.minimum_confidence:
                return calibrated
            return OccupantAssignment("unknown", calibrated.confidence, "NO_OCCUPANT_MATCH", None)

        object_center = (
            (detection.xyxy[0] + detection.xyxy[2]) / 2,
            (detection.xyxy[1] + detection.xyxy[3]) / 2,
        )
        diagonal = max(hypot(frame_width, frame_height), 1.0)
        candidates: list[tuple[float, NormalizedDetection, OccupantAssignment]] = []
        for occupant, assignment in occupants:
            overlap = intersection_over_box(detection.xyxy, occupant.xyxy)
            center = (
                (occupant.xyxy[0] + occupant.xyxy[2]) / 2,
                (occupant.xyxy[1] + occupant.xyxy[3]) / 2,
            )
            distance = hypot(object_center[0] - center[0], object_center[1] - center[1]) / diagonal
            score = 0.65 * overlap + 0.35 * max(0.0, 1.0 - distance * 3.0)
            candidates.append((score, occupant, assignment))
        score, occupant, assignment = max(candidates, key=lambda item: item[0])
        confidence = min(assignment.confidence, score)
        if score < 0.35 or not assignment.known or confidence < self.minimum_confidence:
            return OccupantAssignment(
                "unknown", confidence, "OBJECT_OCCUPANT_LOW_CONFIDENCE", occupant.track_id
            )
        return OccupantAssignment(
            assignment.role, confidence, "OBJECT_TO_OCCUPANT", occupant.track_id
        )

    def _calibrated(
        self,
        detection: NormalizedDetection,
        frame_width: int,
        frame_height: int,
    ) -> OccupantAssignment:
        best_role = "unknown"
        best_overlap = 0.0
        for item in self.calibrated_regions:
            region = OccupantRegion(**item)
            if not region.enabled:
                continue
            overlap = intersection_over_box(
                detection.xyxy,
                normalized_roi_to_pixels(region.normalized_xyxy, frame_width, frame_height),
            )
            if overlap >= region.min_overlap and overlap > best_overlap:
                best_role, best_overlap = region.role, overlap
        if best_role not in self.VALID_ROLES:
            return OccupantAssignment("unknown", 0.0, "INVALID_CALIBRATION", detection.track_id)
        return OccupantAssignment(best_role, best_overlap, "CAMERA_CALIBRATION", detection.track_id)
