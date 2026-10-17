"""Original-resolution behavior ROI diagnostics with verifiable provenance."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from training.dmd.behavior_roi import BehaviorRoiExtractor, Box, RoiStrategy


def frame_sha256(frame: np.ndarray) -> str:
    if frame.ndim not in {2, 3} or not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    header = f"{frame.dtype.str}:{','.join(map(str, frame.shape))}:".encode()
    return hashlib.sha256(header + frame.tobytes()).hexdigest()


@dataclass(frozen=True)
class HighResolutionCrop:
    strategy: str
    frame_sha256: str
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    phone_coverage: float | None
    edge_clipped: bool
    provenance: dict
    image: np.ndarray

    def record(self) -> dict:
        result = asdict(self)
        result.pop("image")
        return result


class HighResolutionRoiPipeline:
    """Crop source pixels without upscaling a cached low-resolution image."""

    def __init__(self, padding: float = 0.12):
        self.extractor = BehaviorRoiExtractor(output_size=None, padding=padding)

    def extract(
        self,
        frame: np.ndarray,
        driver_box: Box,
        strategy: RoiStrategy,
        *,
        source_camera: str,
        phone_box: Box | None = None,
        hand_boxes: tuple[Box, ...] = (),
        face_box: Box | None = None,
        driver_track_id: str | None = None,
        confidence: float = 1.0,
    ) -> list[HighResolutionCrop]:
        digest = frame_sha256(frame)
        height, width = frame.shape[:2]
        crops = self.extractor.extract(
            frame,
            driver_box,
            strategy,
            source_camera=source_camera,
            phone_box=phone_box,
            hand_boxes=hand_boxes,
            face_box=face_box,
            driver_track_id=driver_track_id,
            confidence=confidence,
        )
        results = []
        for crop in crops:
            p = crop.provenance
            if phone_box is None:
                coverage = None
            else:
                x1, y1, x2, y2 = p.crop_box
                ix = max(0.0, min(phone_box.x2, x2) - max(phone_box.x1, x1))
                iy = max(0.0, min(phone_box.y2, y2) - max(phone_box.y1, y1))
                coverage = ix * iy / (phone_box.width * phone_box.height)
            results.append(
                HighResolutionCrop(
                    strategy=strategy.value,
                    frame_sha256=digest,
                    input_dimensions=(width, height),
                    output_dimensions=(crop.image.shape[1], crop.image.shape[0]),
                    phone_coverage=coverage,
                    edge_clipped=any(p.padding_ltrb),
                    provenance=p.as_dict(),
                    image=crop.image,
                )
            )
        return results
