from __future__ import annotations

from dataclasses import dataclass

from backend.ai.detector import NormalizedDetection


@dataclass(frozen=True)
class OccupantRegion:
    role: str
    normalized_xyxy: list[float]
    enabled: bool = True
    min_overlap: float = 0.5


def normalized_roi_to_pixels(roi: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    return x1 * width, y1 * height, x2 * width, y2 * height


def intersection_over_box(box: tuple[float, float, float, float], roi: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    intersection = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return 0.0 if area == 0 else intersection / area


def associate_driver(detection: NormalizedDetection, frame_width: int, frame_height: int, roi: list[float], min_overlap: float = 0.5) -> bool:
    return intersection_over_box(detection.xyxy, normalized_roi_to_pixels(roi, frame_width, frame_height)) >= min_overlap


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
