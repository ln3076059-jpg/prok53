"""Regression tests for V25 action state, timing, and occupant identity fixes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.ai.auxiliary import PoseEvidence
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_checkpoint import CheckpointSafetyError
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import AssociationState, DriverPhoneAssociation
from training.dmd.run_autonomous_v21 import (
    RawVideoBoundaryCollector,
    load_verified_driver_seat,
    verified_driver_track,
)
from training.dmd.run_verified_v24 import (
    OUTPUT,
    StrictTimestampEmbeddingIndex,
    V24ActiveBoundaryReporter,
    load_calibration,
    replay_v24_record,
    replay_v24_records_aligned,
    verify_occupants,
)


def _adapter(mode: str = "V6_HIERARCHICAL") -> ActionInferenceAdapter:
    return ActionInferenceAdapter(
        mode=mode,
        heldout_subject="gC-14",
        causal_mode="DECLARED_LATENCY_CENTERED",
        missing_checkpoint_policy="ERROR",
    )


def test_raw_video_pose_uses_verified_driver_seat_calibration() -> None:
    seat = load_verified_driver_seat()
    assert seat["role"] == "driver"
    pose = PoseEvidence(
        confidence=0.9,
        face_points=((400.0, 100.0),),
        hand_points=((350.0, 250.0),),
        xyxy=(100.0, 50.0, 700.0, 550.0),
    )
    track = verified_driver_track(pose, 832, 612)
    assert track is not None
    assert track.role == "driver"
    assert track.verified is True
    association = DriverPhoneAssociation(minimum_containment=0.35).associate(
        Box(320.0, 220.0, 380.0, 300.0),
        (track,),
        phone_context="HANDHELD_USE",
        phone_confidence=0.75,
        hand_proximity=1.0,
        context_verified=True,
    )
    assert association.state == AssociationState.DRIVER_ASSOCIATED.value


def test_raw_video_pose_without_verified_geometry_fails_closed() -> None:
    missing_box = PoseEvidence(0.9, (), (), None)
    assert verified_driver_track(missing_box, 832, 612) is None
    invalid = PoseEvidence(0.9, (), (), (100.0, 100.0, 100.0, 200.0))
    assert verified_driver_track(invalid, 832, 612) is None
    low_confidence = PoseEvidence(0.24, (), (), (10.0, 10.0, 100.0, 200.0))
    assert verified_driver_track(low_confidence, 832, 612) is None


def test_raw_video_lifecycle_expires_missing_driver_state() -> None:
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.5,
        min_observations=2,
        cooldown_seconds=3.0,
        feature_positive_score=0.18,
        positive_ratio=0.4,
        candidate_threshold=0.18,
        activation_threshold=0.25,
        release_threshold=0.15,
    )
    collector = RawVideoBoundaryCollector(engine)
    for timestamp in (0.0, 0.5, 1.0):
        collector.add(
            Observation(
                timestamp=timestamp,
                class_name="phone",
                confidence=0.6,
                track_id="driver_seat_pose_0",
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="HANDHELD_USE",
                fusion_score=0.6,
            ),
        )
    assert engine.active_events
    collector.add(
        Observation(
            timestamp=4.0,
            class_name="phone",
            confidence=0.0,
            track_id=None,
            occupant_role="UNKNOWN",
            vehicle_context_id="DMD_CABIN",
            fusion_score=0.0,
        ),
    )
    assert not engine.active_events
    for timestamp in (5.0, 5.5, 6.0):
        collector.add(
            Observation(
                timestamp=timestamp,
                class_name="phone",
                confidence=0.6,
                track_id="driver_seat_pose_0",
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="HANDHELD_USE",
                fusion_score=0.6,
            )
        )
    assert [(row["start_seconds"], row["end_seconds"]) for row in collector.intervals] == [
        (0.0, 1.0),
        (5.0, 6.0),
    ]


def test_duplicate_feature_delivery_is_idempotent() -> None:
    adapter = _adapter()
    key = ("gC-14", "s2", "BODY", "driver:FRONT_LEFT")
    embedding = np.zeros(512, dtype=np.float32)
    adapter.observe_embedding(embedding, key, 0.0)
    assert adapter.observe_embedding(embedding, key, 0.0) is None
    assert len(adapter.embedding_buffer) == 1
    with pytest.raises(CheckpointSafetyError, match="OUT_OF_ORDER"):
        adapter.observe_embedding(embedding, key, -0.5)


def test_observe_then_score_does_not_update_state_twice() -> None:
    adapter = _adapter()
    key = ("gC-14", "s2", "BODY", "driver:FRONT_LEFT")
    embedding = np.zeros(512, dtype=np.float32)
    sequences = []
    for index in range(12):
        sequence = adapter.observe_embedding(embedding, key, index * 0.5)
        if sequence is not None:
            scores = adapter.predict_from_sequence(
                sequence,
                pose=None,
                session_id="s2",
                camera_id="BODY",
                occupant_track_id="driver:FRONT_LEFT",
                sequence_timestamps=adapter.current_sequence_timestamps(),
                available_at_timestamp=index * 0.5,
            )
            sequences.append(scores)
    assert len(sequences) == 4
    assert adapter.forward_count == 4
    assert len(adapter.embedding_buffer) == 9
    assert sequences[-1].provenance["score_timestamp"] == 3.5
    assert sequences[-1].provenance["available_at_timestamp"] == 5.5
    assert sequences[-1].provenance["rule_predicted_class"] in {
        "NORMAL",
        "PHONECALL",
        "TEXTING",
        "OTHER_GESTURE",
    }
    assert 0.0 <= sequences[-1].provenance["rule_phonecall"] <= 1.0
    assert 0.0 <= sequences[-1].provenance["rule_texting"] <= 1.0


def test_unverified_calibration_is_rejected_through_loader(tmp_path: Path) -> None:
    calibration = load_calibration(OUTPUT)
    calibration["status"] = "UNVERIFIED_STATIC_CALIBRATION"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="unverified"):
        load_calibration(calibration_path=path)


def test_driver_pose_binding_and_track_id_are_order_invariant() -> None:
    calibration = load_calibration(OUTPUT)
    driver = {
        "confidence": 0.85,
        "xyxy": [100.0, 50.0, 700.0, 550.0],
        "face_points": [[200.0, 100.0]],
        "hand_points": [[250.0, 250.0]],
    }
    passenger = {
        "confidence": 0.9,
        "xyxy": [850.0, 50.0, 1200.0, 550.0],
        "face_points": [[950.0, 100.0]],
        "hand_points": [[900.0, 250.0]],
    }
    for poses in ([passenger, driver], [driver, passenger]):
        _, track, diagnostics = verify_occupants(poses, calibration, [0, 72, 832, 684])
        assert track is not None
        assert track.track_id == "driver:FRONT_LEFT"
        assert poses[diagnostics["driver_pose_index"]]["face_points"] == driver["face_points"]


def test_replay_path_executes_deep_forward_after_warmup(tmp_path: Path) -> None:
    calibration = load_calibration(OUTPUT)
    timestamps = np.arange(12, dtype=np.float64) * 0.5
    embeddings = np.zeros((12, 512), dtype=np.float32)
    cache = tmp_path / "features.npz"
    np.savez(cache, timestamps=timestamps, embeddings=embeddings)
    index = StrictTimestampEmbeddingIndex(cache)
    adapter = _adapter()
    records = []
    for frame, timestamp in enumerate(timestamps):
        records.append({
            "frame_idx": frame,
            "timestamp": float(timestamp),
            "roi_xyxy": [0, 72, 832, 684],
            "poses": [
                {
                    "confidence": 0.9,
                    "xyxy": [100.0, 50.0, 700.0, 550.0],
                    "face_points": [[200.0, 100.0]],
                    "hand_points": [[250.0, 250.0]],
                }
            ],
            "candidates": [],
            "observation": {"timestamp": float(timestamp)},
        })
    replay_rows = replay_v24_records_aligned(
        records,
        adapter,
        calibration,
        {"association_mode": "VERIFIED_SEAT_GEOMETRY_V24", "sample_stride": 1},
        index,
        session_id="s2",
        subject_id="gC-14",
    )
    actions = [detail["action"] for _, detail in replay_rows]
    valid = [action for action in actions if action and action.get("score_valid")]
    assert len(valid) == 4
    assert adapter.forward_count == 4
    assert all(action["available_at_timestamp"] >= action["score_timestamp"] for action in valid)
    assert [action["target_frame_idx"] for action in valid] == [4, 5, 6, 7]
    assert [action["available_at_frame_idx"] for action in valid] == [8, 9, 10, 11]


def test_ambiguous_two_driver_poses_fail_closed() -> None:
    calibration = load_calibration(OUTPUT)
    pose = {"confidence": 0.9, "xyxy": [100.0, 50.0, 700.0, 550.0]}
    occupants, driver, diagnostics = verify_occupants(
        [pose, dict(pose, confidence=0.8)], calibration, [0, 72, 832, 684]
    )
    assert driver is None
    assert "driver" in diagnostics["ambiguous_roles"]
    assert all(not occupant.verified for occupant in occupants)


def test_missing_pose_does_not_break_driver_roi_action_history(tmp_path: Path) -> None:
    calibration = load_calibration(OUTPUT)
    timestamps = np.arange(12, dtype=np.float64) * 0.5
    cache = tmp_path / "features.npz"
    np.savez(cache, timestamps=timestamps, embeddings=np.zeros((12, 512), dtype=np.float32))
    index = StrictTimestampEmbeddingIndex(cache)
    adapter = _adapter()
    records = []
    for frame, timestamp in enumerate(timestamps):
        records.append({
            "frame_idx": frame,
            "timestamp": float(timestamp),
            "roi_xyxy": [0, 72, 832, 684],
            "poses": [],
            "candidates": [],
            "observation": {"timestamp": float(timestamp)},
        })
    replay_rows = replay_v24_records_aligned(
        records,
        adapter,
        calibration,
        {"association_mode": "VERIFIED_SEAT_GEOMETRY_V24", "sample_stride": 1},
        index,
        session_id="s2",
        subject_id="gC-14",
    )
    for observation, _ in replay_rows:
        assert observation.occupant_role == "UNKNOWN"
    assert adapter.forward_count == 4


def test_frame_identity_accepts_timestamp_rounding_without_future_leakage(tmp_path: Path) -> None:
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        timestamps=np.array([0.0, 0.504032258]),
        embeddings=np.zeros((2, 512), dtype=np.float32),
    )
    index = StrictTimestampEmbeddingIndex(cache)
    record = index.get_embedding_record(0.504, expected_index=1)
    assert record is not None
    assert record["feature_index"] == 1
    assert record["identity"] == "FRAME_INDEX_AND_TIMESTAMP"
    assert record["feature_timestamp"] > record["query_timestamp"]
    assert index.get_embedding_record(0.40, expected_index=1) is None


def test_active_boundary_expires_stale_driver_across_unknown_pose_gap() -> None:
    reporter = V24ActiveBoundaryReporter(
        {
            "window_seconds": 2.0,
            "min_positive_seconds": 0.5,
            "min_observations": 2,
            "cooldown_seconds": 3.0,
            "feature_positive_score": 0.18,
            "positive_ratio": 0.4,
            "candidate_threshold": 0.18,
            "activation_threshold": 0.25,
            "release_threshold": 0.15,
        }
    )

    def positive(timestamp: float) -> Observation:
        return Observation(
            timestamp=timestamp,
            class_name="phone",
            confidence=0.6,
            track_id=None,
            occupant_role="driver",
            occupant_id="driver:FRONT_LEFT",
            vehicle_context_id="DMD_CABIN",
            phone_context="HANDHELD_USE",
            fusion_score=0.6,
            evidence_source="TEST_VERIFIED_ASSOCIATION",
        )

    for timestamp in (0.0, 0.5, 1.0):
        reporter.add(positive(timestamp))

    reporter.add(
        Observation(
            timestamp=4.0,
            class_name="phone",
            confidence=0.0,
            track_id=None,
            occupant_role="UNKNOWN",
            vehicle_context_id="DMD_CABIN",
            fusion_score=0.0,
        )
    )
    assert reporter.last_transition["expired_active_count"] == 1

    for timestamp in (5.0, 5.5, 6.0):
        reporter.add(positive(timestamp))

    intervals = reporter.finish(cooldown=0.0)
    assert len(intervals) == 2
    assert intervals[0]["end_seconds"] == 1.0
    assert intervals[1]["start_seconds"] == 5.0
