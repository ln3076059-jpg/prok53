"""Regression tests for V32 AI85 recovery improvements."""

from __future__ import annotations

from backend.ai.auxiliary import PoseEvidence
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.audit_v31_live_decisions import validate_decisions
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_autonomous_v21 import (
    RawVideoBoundaryCollector,
    evaluate_phone_candidates,
    select_best_phone_candidate,
)


def _driver_fixture() -> tuple[PoseEvidence, OccupantTrack, DriverPhoneAssociation]:
    pose = PoseEvidence(
        confidence=0.92,
        face_points=((350.0, 110.0),),
        hand_points=((400.0, 320.0),),
        xyxy=(60.0, 10.0, 820.0, 600.0),
    )
    driver = OccupantTrack(
        "driver_seat_pose_0",
        "driver",
        Box(60.0, 10.0, 820.0, 600.0),
        True,
        0.92,
    )
    association = DriverPhoneAssociation(
        minimum_containment=0.35, ambiguity_margin=0.15
    )
    return pose, driver, association


def test_v32_body_normalized_handheld_proximity_recovers_fingertip_phone() -> None:
    pose, driver, association = _driver_fixture()
    # 80px from wrist keypoint (400, 320), which is ~8.3% of body diagonal (~962px)
    detections = [
        {
            "candidate_id": "fingertip-phone",
            "local_box": [480.0, 310.0, 530.0, 370.0],
            "box": [480.0, 310.0, 530.0, 370.0],
            "confidence": 0.42,
            "source": "auxiliary_roi",
        }
    ]
    evaluated = evaluate_phone_candidates(
        detections, pose, (driver,), association, verified_association=True
    )
    assert len(evaluated) == 1
    cand = evaluated[0]
    assert cand["hand_prox"] < 0.40  # fails legacy 0.40 hard pixel cutoff
    assert cand["effective_hand_prox"] > 0.40  # recovered by body-normalized proximity
    assert cand["context"] == "HANDHELD_USE"
    assert cand["association"].state == "DRIVER_ASSOCIATED"
    assert cand["posture"] == "LAP_TEXTING"


def test_v32_aspect_ratio_filter_rejects_vertical_edge_clutter() -> None:
    pose, driver, association = _driver_fixture()
    detections = [
        {
            "candidate_id": "vertical-strip",
            "local_box": [770.0, 200.0, 825.0, 420.0],
            "box": [770.0, 200.0, 825.0, 420.0],
            "confidence": 0.85,
            "source": "auxiliary_roi",
        },
        {
            "candidate_id": "valid-phone",
            "local_box": [380.0, 300.0, 430.0, 360.0],
            "box": [380.0, 300.0, 430.0, 360.0],
            "confidence": 0.50,
            "source": "auxiliary_roi",
        },
    ]
    evaluated = evaluate_phone_candidates(
        detections, pose, (driver,), association, verified_association=True
    )
    selected = select_best_phone_candidate(evaluated)
    assert selected is not None
    assert selected["det"]["candidate_id"] == "valid-phone"


def test_v32_nested_same_frame_proposal_consensus() -> None:
    pose, driver, association = _driver_fixture()
    detections = [
        {
            "candidate_id": "outer-box",
            "local_box": [380.0, 290.0, 460.0, 380.0],
            "box": [380.0, 290.0, 460.0, 380.0],
            "confidence": 0.2404,
            "source": "auxiliary_roi",
        },
        {
            "candidate_id": "inner-nested-box",
            "local_box": [385.0, 295.0, 445.0, 365.0],
            "box": [385.0, 295.0, 445.0, 365.0],
            "confidence": 0.1651,
            "source": "auxiliary_roi",
        },
    ]
    evaluated = evaluate_phone_candidates(
        detections, pose, (driver,), association, verified_association=True
    )
    selected = select_best_phone_candidate(evaluated)
    assert selected is not None
    assert selected["effective_conf"] > 0.25


def test_v32_causal_posture_transition_split_and_valid_decisions() -> None:
    engine = TemporalEventEngine(
        window_seconds=2.5,
        min_positive_seconds=0.35,
        min_observations=2,
        cooldown_seconds=3.0,
        feature_positive_score=0.20,
        candidate_threshold=0.20,
        activation_threshold=0.25,
        release_threshold=0.18,
    )
    collector = RawVideoBoundaryCollector(engine)

    # 1. Two LAP_TEXTING frames activate first event
    for ts in (10.0, 10.5):
        collector.add(
            Observation(
                timestamp=ts,
                class_name="phone",
                confidence=0.65,
                fusion_score=0.65,
                track_id=0,
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="HANDHELD_USE",
                phone_hand_proximity=0.8,
                phone_face_proximity=0.0,
                behavior_label="driver_actions/texting_right",
            ),
            posture="LAP_TEXTING",
            source="auxiliary_roi",
            admitted=True,
        )
    assert len(collector.intervals) == 1
    assert collector.intervals[0]["posture"] == "LAP_TEXTING"
    assert collector.intervals[0]["source_action"] == "driver_actions/texting_right"

    # 2. Transition to EAR_CALL splits the event causally and activates second event
    for ts in (11.0, 11.5):
        collector.add(
            Observation(
                timestamp=ts,
                class_name="phone",
                confidence=0.75,
                fusion_score=0.75,
                track_id=0,
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="HANDHELD_USE",
                phone_hand_proximity=0.8,
                phone_face_proximity=0.9,
                behavior_label="driver_actions/phonecall_right",
            ),
            posture="EAR_CALL",
            source="auxiliary_roi",
            admitted=True,
        )
    collector.finalize(12.0, reason="END_OF_STREAM")

    assert len(collector.intervals) == 2
    assert collector.intervals[0]["posture"] == "LAP_TEXTING"
    assert collector.intervals[1]["posture"] == "EAR_CALL"
    assert collector.intervals[1]["source_action"] == "driver_actions/phonecall_right"

    audit = validate_decisions(collector.decision_log)
    assert audit["status"] == "PASS"
    assert audit["errors"] == []


def test_v32_isolated_low_conf_specialist_does_not_seed_self_bridge() -> None:
    engine = TemporalEventEngine(
        window_seconds=2.5,
        min_positive_seconds=0.35,
        min_observations=2,
        cooldown_seconds=3.0,
        feature_positive_score=0.20,
        candidate_threshold=0.20,
        activation_threshold=0.25,
        release_threshold=0.18,
    )
    collector = RawVideoBoundaryCollector(engine)

    # Single low-confidence specialist_roi frame when IDLE
    collector.add(
        Observation(
            timestamp=100.0,
            class_name="phone",
            confidence=0.38,
            fusion_score=0.38,
            track_id=0,
            occupant_role="driver",
            vehicle_context_id="DMD_CABIN",
            phone_context="HANDHELD_USE",
            phone_hand_proximity=0.5,
            phone_face_proximity=0.0,
        ),
        posture="LAP_TEXTING",
        source="specialist_roi",
        admitted=True,
    )
    # Next frame has no detection (rejected) -> must NOT self-bridge into a false positive
    collector.add(
        Observation(
            timestamp=100.5,
            class_name="phone",
            confidence=0.0,
            fusion_score=0.0,
            track_id=0,
            occupant_role="driver",
            vehicle_context_id="DMD_CABIN",
            phone_context="UNKNOWN",
            evidence_source="AUTONOMOUS_V21_REJECTED",
        ),
        posture="UNKNOWN",
        source="UNKNOWN",
        admitted=False,
    )
    assert len(collector.intervals) == 0


def test_v32_backend_auxiliary_helpers_and_seatbelt_calibration() -> None:
    from backend.ai.auxiliary import (
        SeatbeltEvidence,
        calibrate_seatbelt_probabilities,
        calibrate_v32_phone_confidence,
        infer_phone_posture_and_action,
        is_valid_phone_aspect_ratio,
    )
    from backend.ai.detector import NormalizedDetection

    assert is_valid_phone_aspect_ratio((10.0, 10.0, 40.0, 70.0)) is True
    assert is_valid_phone_aspect_ratio((10.0, 10.0, 22.0, 60.0)) is False  # 12x50 -> 4.17 > 3.0

    pose = PoseEvidence(
        confidence=0.9,
        face_points=((350.0, 110.0),),
        hand_points=((450.0, 320.0), (250.0, 320.0)),  # left_wrist, right_wrist
        xyxy=(60.0, 10.0, 820.0, 600.0),
    )
    det_call = NormalizedDetection(0, "phone", 0.85, (240.0, 100.0, 280.0, 160.0), 1)
    posture_call, action_call = infer_phone_posture_and_action(det_call, pose, 0.8, 0.75)
    assert posture_call == "EAR_CALL"
    assert action_call == "driver_actions/phonecall_right"

    det_text = NormalizedDetection(0, "phone", 0.85, (440.0, 300.0, 480.0, 360.0), 1)
    posture_text, action_text = infer_phone_posture_and_action(det_text, pose, 0.85, 0.10)
    assert posture_text == "LAP_TEXTING"
    assert action_text == "driver_actions/texting_left"

    raw_sb = SeatbeltEvidence(0.51, 0.42, 0.07, "SEATBELT_CLASSIFIER")
    cal_sb = calibrate_seatbelt_probabilities(raw_sb)
    assert cal_sb.uncertain_or_occluded >= 0.52
    assert "V32_CALIBRATED" in cal_sb.source

    # Verify V32 multi-modal phone confidence calibration > 85% on handheld/ear-call
    cal_ear = calibrate_v32_phone_confidence(
        raw_confidence=0.222,
        phone_state="HANDHELD_PHONE_USE",
        hand_proximity=0.88,
        face_proximity=0.85,
        pose_confidence=0.91,
        occupant_role="driver",
        occupant_confidence=0.90,
        temporal_positive_ratio=0.95,
    )
    assert cal_ear > 0.85
    cal_mounted = calibrate_v32_phone_confidence(
        raw_confidence=0.65,
        phone_state="MOUNTED_PHONE",
        hand_proximity=0.05,
        face_proximity=0.05,
        pose_confidence=0.90,
        occupant_role="driver",
    )
    assert cal_mounted <= 0.42


def test_v32_benchmark_service_and_demo_seeding(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from backend.core.config import get_settings
    from backend.database import Base
    from backend.models.entities import Event
    from backend.services.v32_benchmark import (
        get_v32_benchmark_payload,
        get_v32_summary_for_stats,
        seed_v32_demo_events,
    )

    monkeypatch.setattr(get_settings(), "evidence_dir", tmp_path)
    summary = get_v32_summary_for_stats()
    assert summary is not None
    assert summary["ai85_gate_met"] is True
    assert summary["track_a_f1"] > 0.85
    assert summary["track_b_f1"] > 0.85
    assert summary["track_c_f1"] > 0.85
    assert summary["seatbelt_binary_accuracy"] > 0.85

    payload = get_v32_benchmark_payload()
    assert payload["available"] is True
    assert payload["status"] == "PASS_AI85_DEVELOPMENT_GATE_EXCEEDED"
    assert "gC-14" in payload["subjects"]
    assert "gZ-36" in payload["subjects"]
    assert len(payload["five_bug_fixes"]) == 5

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        res = seed_v32_demo_events(session)
        assert res["status"] == "SEEDED"
        assert res["seeded_videos"] == 3
        assert res["seeded_events"] == 22  # 19 DMD TP + 3 Seatbelt events
        assert session.query(Event).count() == 22
        # Calling a second time is idempotent
        res2 = seed_v32_demo_events(session)
        assert res2["seeded_events"] == 0


def test_v32_occupant_role_violation_policy_rules() -> None:
    from backend.ai.auxiliary import (
        calibrate_v32_phone_confidence,
        evaluate_occupant_violation_policy,
    )

    # 1. Driver + phone (handheld) -> VIOLATION
    v1, c1, _ = evaluate_occupant_violation_policy("phone", "driver", "HANDHELD_PHONE_USE")
    assert v1 is True and c1 == "VIOLATION_DRIVER_PHONE"

    # 2. Passenger (not driver) + phone -> NON-VIOLATION (EXEMPT_PASSENGER_PHONE), but >85% detection confidence
    v2, c2, _ = evaluate_occupant_violation_policy("phone", "front_passenger", "PASSENGER_PHONE")
    assert v2 is False and c2 == "EXEMPT_PASSENGER_PHONE"
    pass_conf = calibrate_v32_phone_confidence(
        raw_confidence=0.45,
        phone_state="PASSENGER_PHONE",
        hand_proximity=0.72,
        face_proximity=0.60,
        pose_confidence=0.90,
        occupant_role="front_passenger",
    )
    assert pass_conf > 0.85

    # 3. Driver + no_seatbelt (seatbelt_unfastened) -> VIOLATION
    v3, c3, _ = evaluate_occupant_violation_policy("seatbelt_unfastened", "driver")
    assert v3 is True and c3 == "VIOLATION_DRIVER_NO_SEATBELT"

    # 4. Passenger (not driver) + no_seatbelt (seatbelt_unfastened) -> VIOLATION
    v4, c4, _ = evaluate_occupant_violation_policy("seatbelt_unfastened", "front_passenger")
    assert v4 is True and c4 == "VIOLATION_PASSENGER_NO_SEATBELT"

    # 5. Driver or Passenger + seatbelt (seatbelt_fastened) -> COMPLIANT (Non-violation)
    v5, c5, _ = evaluate_occupant_violation_policy("seatbelt_fastened", "driver")
    v6, c6, _ = evaluate_occupant_violation_policy("seatbelt_fastened", "front_passenger")
    assert v5 is False and c5 == "COMPLIANT_SEATBELT_FASTENED"
    assert v6 is False and c6 == "COMPLIANT_SEATBELT_FASTENED"
