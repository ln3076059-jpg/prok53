"""Source-resolution tiled phone inference with auditable coordinate merging.

This is a small, dependency-free adapter inspired by SAHI.  It deliberately
does not copy SAHI's evaluation claims or depend on ground-truth metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from training.dmd.autonomous_phone import PhoneDetector, PhoneProposal, local_to_original
from training.dmd.behavior_roi import Box


@dataclass(frozen=True)
class Tile:
    tile_id: str
    box: Box


@dataclass(frozen=True)
class SlicedPhoneResult:
    proposals: tuple[PhoneProposal, ...]
    raw_proposal_count: int
    merged_proposal_count: int
    tiles: tuple[Tile, ...]
    latency_ms: float

    def record(self) -> dict:
        return {
            "proposals": [asdict(item) for item in self.proposals],
            "raw_proposal_count": self.raw_proposal_count,
            "merged_proposal_count": self.merged_proposal_count,
            "tiles": [asdict(item) for item in self.tiles],
            "latency_ms": self.latency_ms,
        }


def _positions(start: int, stop: int, tile_size: int, stride: int) -> list[int]:
    length = stop - start
    if length <= tile_size:
        return [start]
    values = list(range(start, stop - tile_size + 1, stride))
    final = stop - tile_size
    if values[-1] != final:
        values.append(final)
    return values


def make_overlapping_tiles(
    roi: Box,
    frame_width: int,
    frame_height: int,
    *,
    tile_size: tuple[int, int] = (384, 384),
    overlap: float = 0.25,
) -> tuple[Tile, ...]:
    """Cover a source-frame ROI, including its right and bottom boundaries."""
    if frame_width <= 0 or frame_height <= 0 or min(tile_size) <= 0:
        raise ValueError("frame and tile dimensions must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    clipped = roi.clamp(frame_width, frame_height)
    x1, y1 = int(np.floor(clipped.x1)), int(np.floor(clipped.y1))
    x2, y2 = int(np.ceil(clipped.x2)), int(np.ceil(clipped.y2))
    tw, th = min(tile_size[0], x2 - x1), min(tile_size[1], y2 - y1)
    sx, sy = max(1, int(round(tw * (1 - overlap)))), max(1, int(round(th * (1 - overlap))))
    tiles = []
    for row, top in enumerate(_positions(y1, y2, th, sy)):
        for column, left in enumerate(_positions(x1, x2, tw, sx)):
            tiles.append(Tile(f"r{row:02d}c{column:02d}", Box(left, top, left + tw, top + th)))
    return tuple(tiles)


def box_iou(left: Box, right: Box) -> float:
    ix = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    iy = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = ix * iy
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def merge_phone_proposals(
    proposals: list[PhoneProposal], *, iou_threshold: float = 0.50
) -> tuple[PhoneProposal, ...]:
    """Confidence-ordered NMS for a single phone class."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    kept: list[PhoneProposal] = []
    for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
        if all(box_iou(proposal.box, accepted.box) <= iou_threshold for accepted in kept):
            kept.append(proposal)
    return tuple(kept)


class SlicedPhoneDetector:
    """Run one frozen detector on overlapping source-resolution ROI tiles."""

    def __init__(
        self,
        detector: PhoneDetector,
        *,
        tile_size: tuple[int, int] = (384, 384),
        overlap: float = 0.25,
        merge_iou: float = 0.50,
    ):
        self.detector = detector
        self.model_sha256 = detector.model_sha256
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_iou = merge_iou

    def detect_roi(
        self, frame: np.ndarray, roi: Box, confidence_floor: float
    ) -> SlicedPhoneResult:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected HxWx3 source frame")
        if not 0 <= confidence_floor <= 1:
            raise ValueError("confidence floor must be in [0, 1]")
        started = perf_counter()
        height, width = frame.shape[:2]
        tiles = make_overlapping_tiles(
            roi, width, height, tile_size=self.tile_size, overlap=self.overlap
        )
        raw: list[PhoneProposal] = []
        for tile in tiles:
            left, top = int(tile.box.x1), int(tile.box.y1)
            right, bottom = int(tile.box.x2), int(tile.box.y2)
            crop = frame[top:bottom, left:right]
            for proposal in self.detector.detect(crop, confidence_floor):
                if proposal.model_sha256 != self.model_sha256:
                    raise ValueError("proposal model hash mismatch")
                local = proposal.box.clamp(crop.shape[1], crop.shape[0])
                raw.append(
                    PhoneProposal(
                        local_to_original(local, (left, top)),
                        proposal.confidence,
                        f"SLICED:{tile.tile_id}:{proposal.source}",
                        proposal.model_sha256,
                    )
                )
        merged = merge_phone_proposals(raw, iou_threshold=self.merge_iou)
        return SlicedPhoneResult(
            merged, len(raw), len(merged), tiles, (perf_counter() - started) * 1000
        )

    def detect(self, frame: np.ndarray, confidence_floor: float) -> list[PhoneProposal]:
        height, width = frame.shape[:2]
        return list(self.detect_roi(frame, Box(0, 0, width, height), confidence_floor).proposals)
