"""Regression tests for V27 candidate identity and centered-score alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.ai.auxiliary import PoseEvidence
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.driver_phone_association import DriverPhoneAssociation, OccupantTrack
from training.dmd.behavior_roi import Box
from training.dmd.run_autonomous_v21 import (
    AutonomousVideoInferenceEngineV21,
    evaluate_phone_candidates,
    select_best_phone_candidate,
)
from training.dmd.run_verified_v24 import (
    OUTPUT,
    StrictTimestampEmbeddingIndex,
    load_calibration,
    replay_v24_record,
    replay_v24_records_aligned,
)


def _centered_adapter() -> ActionInferenceAdapter:
    return ActionInferenceAdapter(
        mode="V6_HIERARCHICAL",
        heldout_subject="gC-14",
        causal_mode="DECLARED_LATENCY_CENTERED",
        missing_checkpoint_policy="ERROR",
    )


def test_selected_candidate_identity_drives_action_and_event() -> None:
    pose = PoseEvidence(
        confidence=0.9,
        face_points=((250.0, 100.0),),
        hand_points=((250.0, 250.0),),
        xyxy=(50.0, 20.0, 700.0, 560.0),
    )
    driver = OccupantTrack(
        "driver_seat_pose_0", "driver", Box(50.0, 20.0, 700.0, 560.0), True, 0.9
    )
    detections = [
        {
            "candidate_id": "high-confidence-static",
            "local_box": [600.0, 40.0, 650.0, 90.0],
            "box": [600.0, 40.0, 650.0, 90.0],
            "confidence": 0.99,
            "source": "auxiliary_roi",
        },
        {
            "candidate_id": "driver-handheld",
            "local_box": [235.0, 235.0, 270.0, 280.0],
            "box": [235.0, 235.0, 270.0, 280.0],
            "confidence": 0.55,
            "source": "specialist_roi",
        },
    ]
    evaluated = evaluate_phone_candidates(
        detections,
        pose,
        (driver,),
        DriverPhoneAssociation(minimum_containment=0.35, ambiguity_margin=0.15),
        verified_association=True,
    )
    selected = select_best_phone_candidate(evaluated)
    assert selected is not None
    assert selected["det"]["candidate_id"] == "driver-handheld"


def test_force_handheld_context_is_explicit_diagnostic_override() -> None:
    pose = PoseEvidence(
        confidence=0.9,
        face_points=(),
        hand_points=(),
        xyxy=(10.0, 10.0, 500.0, 500.0),
    )
    driver = OccupantTrack(
        "driver_seat_pose_0", "driver", Box(10.0, 10.0, 500.0, 500.0), True, 0.9
    )
    detection = {
        "candidate_id": "phone",
        "local_box": [100.0, 100.0, 150.0, 160.0],
        "box": [100.0, 100.0, 150.0, 160.0],
        "confidence": 0.5,
        "source": "specialist_roi",
    }
    association = DriverPhoneAssociation(minimum_containment=0.35)
    normal = evaluate_phone_candidates(
        [detection], pose, (driver,), association, verified_association=True
    )[0]
    forced = evaluate_phone_candidates(
        [detection],
        pose,
        (driver,),
        association,
        verified_association=True,
        force_handheld_context=True,
    )[0]
    assert normal["context"] == "MOUNTED_OR_STATIC"
    assert normal["association"].state == "UNKNOWN"
    assert forced["context"] == "HANDHELD_USE"
    assert forced["association"].state == "DRIVER_ASSOCIATED"


def test_raw_runner_detector_imgsz_is_configurable() -> None:
    default_engine = AutonomousVideoInferenceEngineV21()
    assert default_engine.detector_imgsz == 640

    resized_engine = AutonomousVideoInferenceEngineV21(detector_imgsz=960)
    assert resized_engine.detector_imgsz == 960

    with pytest.raises(ValueError, match="detector_imgsz must be at least 32"):
        AutonomousVideoInferenceEngineV21(detector_imgsz=16)


def test_raw_runner_rejects_unaligned_centered_checkpoint() -> None:
    with pytest.raises(
        ValueError, match="CENTERED_ACTION_REQUIRES_TARGET_ALIGNED_SESSION_REPLAY"
    ):
        AutonomousVideoInferenceEngineV21(
            action_mode="V6_HIERARCHICAL",
            action_causal_mode="DECLARED_LATENCY_CENTERED",
        )


def test_single_record_api_rejects_centered_score_without_session_queue() -> None:
    calibration = load_calibration(OUTPUT)
    record = {
        "frame_idx": 0,
        "timestamp": 0.0,
        "roi_xyxy": [0, 72, 832, 684],
        "poses": [],
        "candidates": [],
        "observation": {"timestamp": 0.0},
    }
    with pytest.raises(ValueError, match="CENTERED_ACTION_REQUIRES_SESSION_ALIGNMENT"):
        replay_v24_record(
            record,
            _centered_adapter(),
            calibration,
            {"association_mode": "VERIFIED_SEAT_GEOMETRY_V24"},
        )


def test_centered_score_uses_target_frame_candidate(tmp_path: Path) -> None:
    timestamps = np.arange(12, dtype=np.float64) * 0.5
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        timestamps=timestamps,
        embeddings=np.zeros((12, 512), dtype=np.float32),
    )
    records = []
    for frame_index, timestamp in enumerate(timestamps):
        records.append(
            {
                "frame_idx": frame_index,
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
                "candidates": [
                    {
                        "candidate_id": f"candidate-{frame_index}",
                        "box": [235.0, 307.0, 270.0, 352.0],
                        "local_box": [235.0, 235.0, 270.0, 280.0],
                        "confidence": 0.8,
                        "source": "specialist_roi",
                    }
                ],
                "observation": {"timestamp": float(timestamp)},
            }
        )
    replay_rows = replay_v24_records_aligned(
        records,
        _centered_adapter(),
        load_calibration(OUTPUT),
        {"association_mode": "VERIFIED_SEAT_GEOMETRY_V24", "sample_stride": 1},
        StrictTimestampEmbeddingIndex(cache),
        session_id="s2",
        subject_id="gC-14",
    )
    _, target_detail = replay_rows[4]
    action = target_detail["action"]
    assert action["score_valid"] is True
    assert action["target_frame_idx"] == 4
    assert action["available_at_frame_idx"] == 8
    assert action["candidate_id"] == "candidate-4"
    assert action["score_timestamp"] == pytest.approx(2.0)
    assert action["available_at_timestamp"] == pytest.approx(4.0)
