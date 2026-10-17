"""Behavior-centric ROI extraction with coordinate and uncertainty provenance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum

import cv2
import numpy as np


class RoiStrategy(StrEnum):
    FULL_DRIVER = "C0_FULL_DRIVER"
    UPPER_BODY = "C1_UPPER_BODY"
    HAND_PHONE = "C2_HAND_PHONE"
    FACE_HAND = "C3_FACE_HAND"
    MULTI_REGION = "C4_MULTI_REGION"
    DYNAMIC_PADDED = "C5_DYNAMIC_PADDED"


class EvidenceState(StrEnum):
    OBSERVED = "OBSERVED"
    HEURISTIC_FALLBACK = "HEURISTIC_FALLBACK"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def clamp(self, width: int, height: int) -> Box:
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box is outside the image")
        return Box(x1, y1, x2, y2)

    def expand(self, fraction: float) -> Box:
        if fraction < 0:
            raise ValueError("padding fraction cannot be negative")
        dx = self.width * fraction
        dy = self.height * fraction
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def union(self, other: Box) -> Box:
        return Box(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )


@dataclass(frozen=True)
class RoiProvenance:
    strategy: str
    region_name: str
    original_frame_box: tuple[int, int, int, int]
    requested_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    padding_ltrb: tuple[int, int, int, int]
    scale_xy: tuple[float, float]
    confidence: float
    source_camera: str
    driver_track_id: str
    evidence_state: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoiCrop:
    image: np.ndarray
    provenance: RoiProvenance


def _int_box(box: Box) -> tuple[int, int, int, int]:
    return (
        int(np.floor(box.x1)),
        int(np.floor(box.y1)),
        int(np.ceil(box.x2)),
        int(np.ceil(box.y2)),
    )


def _relative_box(driver: Box, left: float, top: float, right: float, bottom: float) -> Box:
    return Box(
        driver.x1 + left * driver.width,
        driver.y1 + top * driver.height,
        driver.x1 + right * driver.width,
        driver.y1 + bottom * driver.height,
    )


def _union(boxes: Iterable[Box]) -> Box | None:
    iterator = iter(boxes)
    try:
        result = next(iterator)
    except StopIteration:
        return None
    for box in iterator:
        result = result.union(box)
    return result


class BehaviorRoiExtractor:
    """Extract fixed or evidence-driven regions without inventing missing evidence."""

    def __init__(self, output_size: tuple[int, int] = (64, 64), padding: float = 0.12):
        if min(output_size) <= 0:
            raise ValueError("output size must be positive")
        if not 0 <= padding <= 1:
            raise ValueError("padding must be in [0, 1]")
        self.output_size = output_size
        self.padding = padding

    def extract(
        self,
        frame: np.ndarray,
        driver_box: Box,
        strategy: RoiStrategy,
        *,
        source_camera: str,
        driver_track_id: str | None = None,
        phone_box: Box | None = None,
        hand_boxes: tuple[Box, ...] = (),
        face_box: Box | None = None,
        confidence: float = 1.0,
    ) -> list[RoiCrop]:
        if frame.ndim not in {2, 3}:
            raise ValueError("frame must be HxW or HxWxC")
        height, width = frame.shape[:2]
        driver = driver_box.clamp(width, height)
        regions: list[tuple[str, Box, EvidenceState, tuple[str, ...]]] = []
        if strategy == RoiStrategy.FULL_DRIVER:
            regions.append(("driver", driver, EvidenceState.OBSERVED, ("DRIVER_ROI",)))
        elif strategy == RoiStrategy.UPPER_BODY:
            regions.append(
                (
                    "upper_body",
                    _relative_box(driver, 0.05, 0.0, 0.95, 0.76),
                    EvidenceState.HEURISTIC_FALLBACK,
                    ("CAMERA_NORMALIZED_UPPER_BODY",),
                )
            )
        elif strategy == RoiStrategy.HAND_PHONE:
            evidence = tuple(hand_boxes) + ((phone_box,) if phone_box else ())
            observed = _union(evidence)
            if observed:
                regions.append(
                    (
                        "hand_phone",
                        observed.expand(self.padding),
                        EvidenceState.OBSERVED,
                        ("HAND_OR_PHONE_BOX",),
                    )
                )
            else:
                regions.append(
                    (
                        "hand_phone_fallback",
                        _relative_box(driver, 0.03, 0.38, 0.97, 1.0),
                        EvidenceState.UNKNOWN,
                        ("NO_HAND_OR_PHONE_BOX", "CAMERA_NORMALIZED_FALLBACK"),
                    )
                )
        elif strategy == RoiStrategy.FACE_HAND:
            evidence = tuple(hand_boxes) + ((face_box,) if face_box else ())
            observed = _union(evidence)
            if observed:
                regions.append(
                    (
                        "face_hand",
                        observed.expand(self.padding),
                        EvidenceState.OBSERVED,
                        ("FACE_OR_HAND_BOX",),
                    )
                )
            else:
                regions.append(
                    (
                        "face_hand_fallback",
                        _relative_box(driver, 0.03, 0.0, 0.97, 0.82),
                        EvidenceState.UNKNOWN,
                        ("NO_FACE_OR_HAND_BOX", "CAMERA_NORMALIZED_FALLBACK"),
                    )
                )
        elif strategy == RoiStrategy.MULTI_REGION:
            regions.extend(
                [
                    (
                        "face",
                        _relative_box(driver, 0.20, 0.0, 0.80, 0.42),
                        EvidenceState.UNKNOWN,
                        ("FIXED_FACE_ZONE",),
                    ),
                    (
                        "torso",
                        _relative_box(driver, 0.12, 0.25, 0.88, 0.86),
                        EvidenceState.UNKNOWN,
                        ("FIXED_TORSO_ZONE",),
                    ),
                    (
                        "left_hand",
                        _relative_box(driver, 0.0, 0.34, 0.58, 1.0),
                        EvidenceState.UNKNOWN,
                        ("FIXED_LEFT_HAND_ZONE",),
                    ),
                    (
                        "right_hand",
                        _relative_box(driver, 0.42, 0.34, 1.0, 1.0),
                        EvidenceState.UNKNOWN,
                        ("FIXED_RIGHT_HAND_ZONE",),
                    ),
                ]
            )
        elif strategy == RoiStrategy.DYNAMIC_PADDED:
            observed = _union(
                tuple(hand_boxes)
                + ((phone_box,) if phone_box else ())
                + ((face_box,) if face_box else ())
            )
            if observed:
                regions.append(
                    (
                        "dynamic_evidence",
                        observed.expand(self.padding),
                        EvidenceState.OBSERVED,
                        ("DYNAMIC_EVIDENCE_UNION", "BOUNDARY_PROTECTED"),
                    )
                )
            else:
                regions.append(
                    (
                        "dynamic_fallback",
                        driver.expand(self.padding),
                        EvidenceState.UNKNOWN,
                        ("NO_DYNAMIC_EVIDENCE", "PADDED_DRIVER_FALLBACK"),
                    )
                )
        else:
            raise ValueError(f"unsupported strategy: {strategy}")
        return [
            self._crop(
                frame,
                driver,
                requested,
                strategy,
                region_name,
                source_camera,
                driver_track_id,
                confidence,
                evidence_state,
                reasons,
            )
            for region_name, requested, evidence_state, reasons in regions
        ]

    def _crop(
        self,
        frame: np.ndarray,
        driver: Box,
        requested: Box,
        strategy: RoiStrategy,
        region_name: str,
        source_camera: str,
        driver_track_id: str | None,
        confidence: float,
        evidence_state: EvidenceState,
        reasons: tuple[str, ...],
    ) -> RoiCrop:
        height, width = frame.shape[:2]
        rx1, ry1, rx2, ry2 = _int_box(requested)
        left = max(0, -rx1)
        top = max(0, -ry1)
        right = max(0, rx2 - width)
        bottom = max(0, ry2 - height)
        x1, y1 = max(0, rx1), max(0, ry1)
        x2, y2 = min(width, rx2), min(height, ry2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("requested ROI does not overlap the frame")
        crop = frame[y1:y2, x1:x2]
        if any((left, top, right, bottom)):
            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right, cv2.BORDER_REPLICATE
            )
        output_width, output_height = self.output_size
        resized = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_AREA)
        provenance = RoiProvenance(
            strategy=strategy.value,
            region_name=region_name,
            original_frame_box=_int_box(driver),
            requested_box=(rx1, ry1, rx2, ry2),
            crop_box=(x1, y1, x2, y2),
            padding_ltrb=(left, top, right, bottom),
            scale_xy=(output_width / crop.shape[1], output_height / crop.shape[0]),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            source_camera=source_camera,
            driver_track_id=driver_track_id or "UNKNOWN",
            evidence_state=evidence_state.value,
            reason_codes=reasons,
        )
        return RoiCrop(image=resized, provenance=provenance)
