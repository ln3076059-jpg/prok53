from __future__ import annotations

import numpy as np
import pytest

from training.dmd.autonomous_phone import AutonomousPhonePipeline, DriverRegion, PhoneProposal
from training.dmd.behavior_roi import Box
from training.dmd.sliced_phone import (
    SlicedPhoneDetector,
    make_overlapping_tiles,
    merge_phone_proposals,
)


class FixedDetector:
    model_sha256 = "a" * 64

    def __init__(self, proposal: PhoneProposal | None = None):
        self.proposal = proposal

    def detect(self, frame: np.ndarray, confidence_floor: float) -> list[PhoneProposal]:
        return [self.proposal] if self.proposal else []


def proposal(box: Box, confidence: float = 0.8) -> PhoneProposal:
    return PhoneProposal(box, confidence, "FIXTURE", FixedDetector.model_sha256)


def test_tiles_cover_right_and_bottom_roi_boundaries() -> None:
    tiles = make_overlapping_tiles(Box(100, 50, 900, 650), 1000, 700, tile_size=(384, 320))
    assert min(tile.box.x1 for tile in tiles) == 100
    assert min(tile.box.y1 for tile in tiles) == 50
    assert max(tile.box.x2 for tile in tiles) == 900
    assert max(tile.box.y2 for tile in tiles) == 650


def test_tile_local_box_maps_to_original_coordinates() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = SlicedPhoneDetector(
        FixedDetector(proposal(Box(5, 7, 25, 37))), tile_size=(100, 100), overlap=0
    )
    result = detector.detect_roi(frame, Box(100, 50, 200, 150), 0.1)
    assert result.proposals[0].box == Box(105, 57, 125, 87)
    assert result.proposals[0].source.startswith("SLICED:r00c00:")


def test_overlapping_tile_duplicates_are_suppressed() -> None:
    items = [proposal(Box(10, 10, 30, 30), 0.7), proposal(Box(11, 11, 31, 31), 0.9)]
    merged = merge_phone_proposals(items, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.9


def test_empty_tiles_remain_empty_and_do_not_imply_safe() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = SlicedPhoneDetector(FixedDetector(), tile_size=(80, 80)).detect_roi(
        frame, Box(0, 0, 160, 120), 0.1
    )
    assert result.proposals == ()
    assert result.raw_proposal_count == result.merged_proposal_count == 0


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_overlapping_tiles(Box(0, 0, 10, 10), 10, 10, overlap=1.0)


def test_autonomous_pipeline_uses_driver_roi_for_sliced_detector() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector = SlicedPhoneDetector(FixedDetector(), tile_size=(100, 100), overlap=0.25)
    result = AutonomousPhonePipeline(detector).process(
        frame,
        {"camera": "BODY"},
        DriverRegion(Box(40, 20, 240, 220), "driver-1", "driver", 0.9, True, "FIXTURE"),
    )
    assert result["status"] == "NO_PHONE_PROPOSAL"
    assert result["detector_diagnostics"]["mode"] == "SLICED_DRIVER_ROI"
    assert result["detector_diagnostics"]["tile_count"] > 1
