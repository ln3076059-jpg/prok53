from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CabinRegion:
    """A localized cabin/windshield region inside a vehicle crop."""

    xyxy: tuple[float, float, float, float] | None
    confidence: float
    method: str
    status: str
    reason: str | None = None

    @property
    def known(self) -> bool:
        return self.status == "KNOWN_CABIN" and self.xyxy is not None


class CabinLocalizer:
    """Fail-closed cabin localizer with replaceable localization strategies.

    Strategy order is custom detector, approved camera calibration, then an explicitly
    low-confidence geometry proposal. Geometry is never silently presented as a known cabin.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        weights = self.config.get("weights")
        self.weights = Path(weights) if weights else None
        self.minimum_confidence = float(self.config.get("minimum_confidence", 0.60))
        self.geometry_confidence = float(self.config.get("geometry_confidence", 0.45))
        self._model = None

    @property
    def detector_available(self) -> bool:
        return bool(self.weights and self.weights.is_file())

    def _load(self) -> None:
        if self.detector_available and self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))

    def locate(
        self,
        vehicle_crop: np.ndarray,
        vehicle_class: str,
        camera_calibration: dict | None = None,
    ) -> CabinRegion:
        if vehicle_crop.size == 0:
            return CabinRegion(None, 0.0, "NONE", "UNKNOWN_CABIN", "empty vehicle crop")

        detected = self._detect(vehicle_crop)
        if detected is not None:
            return self._gate(detected)

        calibrated = self._calibrated(vehicle_crop, camera_calibration)
        if calibrated is not None:
            return self._gate(calibrated)

        proposed = self._geometry(vehicle_crop, vehicle_class)
        if proposed is None:
            return CabinRegion(
                None,
                0.0,
                "GEOMETRY_FALLBACK",
                "UNKNOWN_CABIN",
                f"no cabin geometry policy for vehicle class {vehicle_class}",
            )
        return self._gate(proposed)

    def _detect(self, crop: np.ndarray) -> CabinRegion | None:
        self._load()
        if self._model is None:
            return None
        result = self._model.predict(crop, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return CabinRegion(
                None, 0.0, "CUSTOM_WINDSHIELD_DETECTOR", "UNKNOWN_CABIN", "no windshield detected"
            )
        rows = list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True))
        xyxy, confidence = max(rows, key=lambda row: float(row[1]))
        return CabinRegion(
            tuple(float(value) for value in xyxy),
            float(confidence),
            "CUSTOM_WINDSHIELD_DETECTOR",
            "KNOWN_CABIN",
        )

    def _calibrated(self, crop: np.ndarray, calibration: dict | None) -> CabinRegion | None:
        if not calibration or not calibration.get("approved", False):
            return None
        roi = calibration.get("normalized_xyxy")
        if not isinstance(roi, list) or len(roi) != 4:
            return CabinRegion(
                None, 0.0, "CAMERA_CALIBRATION", "UNKNOWN_CABIN", "invalid calibrated ROI"
            )
        height, width = crop.shape[:2]
        x1, y1, x2, y2 = (
            float(roi[0]) * width,
            float(roi[1]) * height,
            float(roi[2]) * width,
            float(roi[3]) * height,
        )
        return CabinRegion(
            (x1, y1, x2, y2),
            float(calibration.get("confidence", 1.0)),
            "CAMERA_CALIBRATION",
            "KNOWN_CABIN",
        )

    def _geometry(self, crop: np.ndarray, vehicle_class: str) -> CabinRegion | None:
        # These are proposals only. Their default confidence is deliberately below the
        # production gate until a camera-specific validation raises it explicitly.
        policies = self.config.get("geometry_rois", {})
        roi = policies.get(vehicle_class) or policies.get("default")
        if not roi:
            return None
        height, width = crop.shape[:2]
        return CabinRegion(
            (
                float(roi[0]) * width,
                float(roi[1]) * height,
                float(roi[2]) * width,
                float(roi[3]) * height,
            ),
            self.geometry_confidence,
            "GEOMETRY_FALLBACK",
            "KNOWN_CABIN",
        )

    def _gate(self, region: CabinRegion) -> CabinRegion:
        if region.xyxy is None or region.confidence < self.minimum_confidence:
            return CabinRegion(
                region.xyxy,
                region.confidence,
                region.method,
                "UNKNOWN_CABIN",
                region.reason or f"confidence below {self.minimum_confidence:.2f}",
            )
        x1, y1, x2, y2 = region.xyxy
        if x2 <= x1 or y2 <= y1:
            return CabinRegion(None, 0.0, region.method, "UNKNOWN_CABIN", "invalid cabin bounds")
        return region


def crop_cabin(vehicle_crop: np.ndarray, region: CabinRegion) -> tuple[np.ndarray, tuple[int, int]]:
    if not region.known or region.xyxy is None:
        return vehicle_crop[0:0, 0:0], (0, 0)
    height, width = vehicle_crop.shape[:2]
    x1, y1, x2, y2 = region.xyxy
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(x2)), min(height, int(y2))
    if right <= left or bottom <= top:
        return vehicle_crop[0:0, 0:0], (0, 0)
    return vehicle_crop[top:bottom, left:right], (left, top)
