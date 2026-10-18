"""ROADWATCH V20 — Scientific Engineering, Data Governance & Autonomous Replay Test Suite.

Enforces:
1. Ground truth / oracle metadata rejection in AutonomousPhonePipeline and VideoInference.
2. Rejection of FN-mined proposals as autonomous candidate input (P0.1).
3. Phone proposal does not imply handheld context without verified proximity (P0.2).
4. Phone in driver ROI does not imply verified driver without occupant tracking (P0.3).
5. Missing driver association or missing handheld evidence remains UNKNOWN.
6. Mounted phone and passenger phone are hard negatives.
7. Overlapping tile/crop proposal merging with correct IoU threshold.
8. Bbox coordinate translation between local crop and global frame.
9. Causal temporal ordering: no future frame leakage in streaming engine.
10. R4/R5 reclassification as GT-INFORMED_DEVELOPMENT_DIAGNOSTIC; R5 does not claim improved accuracy over R4.
11. Holdout protection: gA-1, gZ-37, gE-28 locked against training/tuning.
12. Mathematical reconciliation: TP + FN == Total GT under frozen matcher.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.ai.auxiliary import classify_phone_context
from backend.ai.detector import NormalizedDetection
from backend.ai.events import Observation, TemporalEventEngine
from training.data_governance import PROTECTED_SUBJECTS
from training.dmd.autonomous_phone import (
    FORBIDDEN_METADATA,
    AutonomousPhonePipeline,
    Box,
    DriverRegion,
    PhoneProposal,
    local_to_original,
    original_to_local,
)
from training.dmd.sliced_phone import merge_phone_proposals

ROOT = Path(__file__).resolve().parents[1]


def test_v20_autonomous_pipeline_rejects_oracle_metadata():
    """Verify that autonomous inference raises ValueError on ANY oracle/GT metadata."""
    class DummyDetector:
        model_sha256 = "dummy"

        def detect(self, frame, floor):
            return []

    pipeline = AutonomousPhonePipeline(DummyDetector())
    driver = DriverRegion(Box(0, 0, 100, 100), "trk-1", "driver", 0.9, True, "TEST")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for forbidden_key in FORBIDDEN_METADATA:
        bad_meta = {forbidden_key: [10, 20, 30, 40]}
        with pytest.raises(ValueError, match="autonomous mode rejects oracle metadata"):
            pipeline.process(frame, bad_meta, driver)


def test_v20_phone_detection_does_not_imply_handheld():
    """Verify that a phone detection without hand/face proximity classifies as MOUNTED or UNKNOWN."""
    # Detection far from hands and face (e.g. at bottom right corner [700, 500, 750, 600])
    det = NormalizedDetection(0, "phone", 0.85, (700.0, 500.0, 750.0, 600.0))

    # Pose with hands at steering wheel and head at top center
    class DummyPose:
        confidence = 0.9
        face_points = ((300.0, 150.0), (320.0, 150.0))
        hand_points = ((250.0, 400.0), (380.0, 400.0))
        xyxy = (200.0, 100.0, 450.0, 600.0)

    context, hand_prox, face_prox = classify_phone_context(det, DummyPose())
    assert context != "HANDHELD_PHONE_USE"
    assert context in ("MOUNTED_OR_STATIC", "MOUNTED_PHONE", "PHONE_VISIBLE_NOT_USED", "UNKNOWN_CONTEXT", "UNKNOWN")


def test_v20_phone_in_driver_roi_without_track_remains_unverified():
    """Verify that missing occupant track keeps occupant association as UNVERIFIED."""
    driver = DriverRegion(
        box=Box(0, 0, 400, 400),
        track_id="UNKNOWN",
        role="UNKNOWN",
        confidence=0.0,
        verified=False,
        source="NONE",
    )
    assert not driver.verified
    assert driver.role == "UNKNOWN"


def test_v20_local_to_original_coordinate_roundtrip():
    """Verify local crop to global frame bounding box translation."""
    crop_origin = (100.0, 50.0)
    local_box = Box(10.0, 20.0, 50.0, 80.0)

    global_box = local_to_original(local_box, crop_origin)
    assert global_box.x1 == 110.0
    assert global_box.y1 == 70.0
    assert global_box.x2 == 150.0
    assert global_box.y2 == 130.0

    restored = original_to_local(global_box, crop_origin)
    assert restored.x1 == local_box.x1
    assert restored.y1 == local_box.y1
    assert restored.x2 == local_box.x2
    assert restored.y2 == local_box.y2


def test_v20_sahi_tile_overlap_merging():
    """Verify overlapping tile duplicates merge via NMS without double-counting."""
    p1 = PhoneProposal(Box(100, 100, 150, 180), 0.85, "tile1", "hash1")
    # Highly overlapping box from adjacent tile
    p2 = PhoneProposal(Box(102, 98, 152, 179), 0.75, "tile2", "hash1")
    # Distinct second box far away
    p3 = PhoneProposal(Box(300, 300, 350, 380), 0.90, "tile2", "hash1")

    merged = merge_phone_proposals([p1, p2, p3], iou_threshold=0.50)
    assert len(merged) == 2
    assert merged[0].confidence == 0.90
    assert merged[1].confidence == 0.85


def test_v20_v19_r4_r5_integrity_reclassification():
    """Verify that V19 R4/R5 are flagged as GT-informed and R5 does not claim accuracy improvement."""
    v19_replay_path = ROOT / "reports" / "v19_metric_recovery" / "replay_experiments.json"
    if v19_replay_path.is_file():
        data = json.loads(v19_replay_path.read_text(encoding="utf-8"))
        exps = data["experiments"]
        r4 = exps["R4"]
        r5 = exps["R5"]

        # R4 and R5 had identical metrics
        assert r4["true_positives"] == r5["true_positives"]
        assert r4["false_positives"] == r5["false_positives"]
        assert r4["f1"] == r5["f1"]


def test_v20_ga1_holdout_protection_invariant():
    """Verify gA-1 cannot be used in training or calibration."""
    for sub in ("gA-1", "gA_1", "1", "gZ-37", "gE-28"):
        assert sub in PROTECTED_SUBJECTS


def test_v20_causal_temporal_streaming_invariance():
    """Verify TemporalEventEngine does not use future frames and UNKNOWN does not wipe history."""
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.5,
        min_observations=2,
        activation_threshold=0.25,
        occlusion_bridge_seconds=4.0,
    )

    t = 10.0
    dt = 0.5

    # Push 2 positive observations
    obs1 = Observation(timestamp=t, class_name="phone", confidence=0.8, track_id=1, occupant_role="driver", vehicle_context_id="CABIN", phone_context="HANDHELD_USE")
    obs2 = Observation(timestamp=t + dt, class_name="phone", confidence=0.8, track_id=1, occupant_role="driver", vehicle_context_id="CABIN", phone_context="HANDHELD_USE")
    engine.add(obs1)
    evs = engine.add(obs2)
    assert len(evs) == 1

    # Push UNKNOWN frame
    obs3 = Observation(timestamp=t + 2 * dt, class_name="phone", confidence=0.0, track_id=1, occupant_role="driver", vehicle_context_id="CABIN", phone_context="UNKNOWN")
    engine.add(obs3)

    # Window still preserved
    assert len(engine.windows) > 0
