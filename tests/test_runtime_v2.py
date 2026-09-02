import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.ai.association import OccupantAssociator
from backend.ai.auxiliary import SeatbeltEvidence, VehicleRegion
from backend.ai.cabin import CabinLocalizer, CabinRegion, crop_cabin
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from backend.ai.evidence import EventEvidenceBuffer, evidence_package_sha256
from backend.ai.fusion import FusionFeatures, LogisticFusion
from backend.ai.tracking import WithinVehicleTracker
from backend.api import routes as api_routes
from backend.api.routes import review_event
from backend.database import Base
from backend.models.entities import (
    AnalysisJob,
    Event,
    EventType,
    Evidence,
    InputScope,
    ReviewStatus,
    User,
    Video,
)
from backend.schemas import ReviewRequest
from backend.services.video_analyzer import VideoAnalyzer
from training.common import sha256_file
from training.evaluate_associations import evaluate as evaluate_associations
from training.evaluate_events import evaluate
from training.freeze_external_test import freeze_external_test


def engine(**overrides):
    options = {
        "window_seconds": 2,
        "min_positive_seconds": 1,
        "min_observations": 2,
        "cooldown_seconds": 5,
    }
    options.update(overrides)
    return TemporalEventEngine(**options)


def test_motorcycle_never_emits_no_seatbelt():
    runtime = engine()
    runtime.add(
        Observation(0, "seatbelt_unfastened", 0.9, 1, "driver", "moto-1", vehicle_type="motorcycle")
    )
    assert (
        runtime.add(
            Observation(
                1.1, "seatbelt_unfastened", 0.9, 1, "driver", "moto-1", vehicle_type="motorcycle"
            )
        )
        == []
    )


def test_passenger_phone_is_not_an_event():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 2, "front_passenger", "car-1", "HANDHELD_USE"))
    assert (
        runtime.add(Observation(1.1, "phone", 0.9, 2, "front_passenger", "car-1", "HANDHELD_USE"))
        == []
    )


def test_persistent_driver_handheld_phone_emits_candidate():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 3, "driver", "car-1", "HANDHELD_USE"))
    events = runtime.add(Observation(1.1, "phone", 0.9, 3, "driver", "car-1", "HANDHELD_USE"))
    assert [(item.event_type, item.review_status) for item in events] == [("PHONE", "PENDING")]


def test_mounted_phone_is_not_an_event():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 4, "driver", "car-1", "MOUNTED_OR_STATIC"))
    assert (
        runtime.add(Observation(1.1, "phone", 0.9, 4, "driver", "car-1", "MOUNTED_OR_STATIC")) == []
    )


def test_unknown_phone_context_and_unknown_occupant_need_review():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 5, "unknown", "car-1", "UNKNOWN_PHONE_CONTEXT"))
    events = runtime.add(
        Observation(1.1, "phone", 0.9, 5, "unknown", "car-1", "UNKNOWN_PHONE_CONTEXT")
    )
    assert events[0].review_status == "NEEDS_REVIEW"


def test_outside_vehicle_and_uncertain_belt_do_not_emit():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 6, "driver", None, "HANDHELD_USE"))
    assert runtime.add(Observation(1.1, "phone", 0.9, 6, "driver", None, "HANDHELD_USE")) == []
    runtime.add(Observation(0, "seatbelt_uncertain", 0.9, 7, "driver", "car-1"))
    assert runtime.add(Observation(1.1, "seatbelt_uncertain", 0.9, 7, "driver", "car-1")) == []


def test_passenger_unfastened_emits_and_classifier_conflict_needs_review():
    runtime = engine()
    runtime.add(
        Observation(
            0,
            "seatbelt_unfastened",
            0.9,
            8,
            "front_passenger",
            "car-1",
            evidence_source="DETECTOR_CLASSIFIER_CONFLICT",
        )
    )
    events = runtime.add(
        Observation(
            1.1,
            "seatbelt_unfastened",
            0.9,
            8,
            "front_passenger",
            "car-1",
            evidence_source="DETECTOR_CLASSIFIER_CONFLICT",
        )
    )
    assert events[0].event_type == "NO_SEATBELT"
    assert events[0].review_status == "NEEDS_REVIEW"


def test_vehicle_loss_resets_temporal_state():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 9, "driver", "car-1", "HANDHELD_USE"))
    runtime.reset_vehicle("car-1")
    assert runtime.add(Observation(1.1, "phone", 0.9, 9, "driver", "car-1", "HANDHELD_USE")) == []


def test_track_switch_does_not_merge_phone_identity():
    runtime = engine()
    runtime.add(Observation(0, "phone", 0.9, 10, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(1.1, "phone", 0.9, 11, "driver", "car-1", "HANDHELD_USE")) == []


def test_event_cooldown_blocks_duplicate_activation():
    runtime = engine(cooldown_seconds=5)
    runtime.add(Observation(0, "phone", 0.9, 12, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(1.1, "phone", 0.9, 12, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(1.5, "phone", 0.9, 12, "driver", "car-1", "HANDHELD_USE")) == []


def test_hysteresis_blocks_continuous_duplicates_until_explicit_release():
    runtime = engine(cooldown_seconds=2, release_threshold=0.35)
    runtime.add(Observation(0, "phone", 0.9, 13, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(1.1, "phone", 0.9, 13, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(5.0, "phone", 0.9, 13, "driver", "car-1", "HANDHELD_USE")) == []
    assert (
        runtime.add(Observation(5.1, "phone", 0.9, 13, "driver", "car-1", "MOUNTED_OR_STATIC"))
        == []
    )
    runtime.add(Observation(6.0, "phone", 0.9, 13, "driver", "car-1", "HANDHELD_USE"))
    assert runtime.add(Observation(7.1, "phone", 0.9, 13, "driver", "car-1", "HANDHELD_USE"))


def test_cabin_localization_failure_is_explicit():
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    localizer = CabinLocalizer(
        {
            "minimum_confidence": 0.6,
            "geometry_confidence": 0.45,
            "geometry_rois": {"car": [0.1, 0.1, 0.9, 0.6]},
        }
    )
    result = localizer.locate(crop, "car")
    assert result.status == "UNKNOWN_CABIN"
    assert not result.known


def test_right_hand_drive_and_unknown_geometry_are_not_forced_to_driver():
    associator = OccupantAssociator(driver_side="right", minimum_confidence=0.55)
    right = NormalizedDetection(0, "occupant_upper_body", 0.9, (62, 20, 92, 70), 1)
    center = NormalizedDetection(0, "occupant_upper_body", 0.9, (42, 20, 58, 70), 2)
    assert associator.assign_upper_body(right, 100, 100).role == "driver"
    assert associator.assign_upper_body(center, 100, 100).role == "unknown"


def test_track_by_detection_uses_geometry_and_preserves_identity():
    tracker = WithinVehicleTracker(max_gap_frames=3)
    first = tracker.update(
        "car-1", 0, [NormalizedDetection(0, "phone", 0.9, (10, 10, 20, 20))], 100, 100
    )[0]
    second = tracker.update(
        "car-1", 1, [NormalizedDetection(0, "phone", 0.9, (11, 11, 21, 21))], 100, 100
    )[0]
    distant = tracker.update(
        "car-1", 1, [NormalizedDetection(0, "phone", 0.9, (80, 80, 90, 90))], 100, 100
    )[0]
    assert first.track_id == second.track_id
    assert distant.track_id != first.track_id


def test_evidence_buffer_generates_images_clip_trace_and_hashes(tmp_path):
    frame = np.full((48, 64, 3), 90, dtype=np.uint8)
    buffer = EventEvidenceBuffer(fps=2, pre_seconds=1, post_seconds=0.5)
    buffer.add_frame(0.0, frame)
    buffer.add_frame(0.5, frame)
    buffer.register("event-1", tmp_path, 0.5, frame, frame, {"vehicle_type": "car"})
    artifacts = buffer.add_frame(1.0, frame)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.original_path.is_file()
    assert artifact.annotated_path.is_file()
    assert artifact.clip_path is not None and artifact.clip_path.is_file()
    trace = json.loads(artifact.trace_path.read_text(encoding="utf-8"))
    assert trace["evidence_status"] == "COMPLETE"
    assert set(trace["files"]) >= {
        "original_keyframe.jpg",
        "annotated_keyframe.jpg",
        "evidence.mp4",
    }


def test_fusion_fallback_status_is_operator_visible_and_can_fail_closed():
    fallback = LogisticFusion(None, threshold=0.5)
    assert fallback.status()["fusion_mode"] == "RULE_FALLBACK"
    required = LogisticFusion(None, threshold=0.5, require_calibrated=True)
    result = required.predict(FusionFeatures(detector_confidence=1.0, vehicle_context=1.0))
    assert result.decision is False
    assert result.source == "CALIBRATED_MODEL_REQUIRED"
    assert required.status()["fusion_fail_closed"] is True


def test_synthetic_integration_vehicle_to_event_chain():
    vehicle_crop = np.zeros((120, 200, 3), dtype=np.uint8)
    localizer = CabinLocalizer({"minimum_confidence": 0.6})
    cabin_region = localizer.locate(
        vehicle_crop,
        "car",
        {"approved": True, "normalized_xyxy": [0.1, 0.1, 0.9, 0.8], "confidence": 0.95},
    )
    cabin, _ = crop_cabin(vehicle_crop, cabin_region)
    tracker = WithinVehicleTracker()
    occupant = NormalizedDetection(0, "occupant_upper_body", 0.9, (95, 15, 145, 80))
    phone = NormalizedDetection(0, "phone", 0.9, (115, 35, 128, 55))
    tracked = tracker.update("vehicle-1", 0, [occupant, phone], cabin.shape[1], cabin.shape[0])
    associator = OccupantAssociator(driver_side="right")
    tracked_occupant = next(item for item in tracked if item.class_name == "occupant_upper_body")
    tracked_phone = next(item for item in tracked if item.class_name == "phone")
    role = associator.assign_upper_body(tracked_occupant, cabin.shape[1], cabin.shape[0])
    phone_role = associator.assign_object(
        tracked_phone, [(tracked_occupant, role)], cabin.shape[1], cabin.shape[0]
    )
    runtime = engine()
    runtime.add(
        Observation(
            0,
            "phone",
            0.9,
            tracked_phone.track_id,
            phone_role.role,
            "vehicle-1",
            "HANDHELD_USE",
            vehicle_type="car",
        )
    )
    events = runtime.add(
        Observation(
            1.1,
            "phone",
            0.9,
            tracked_phone.track_id,
            phone_role.role,
            "vehicle-1",
            "HANDHELD_USE",
            vehicle_type="car",
        )
    )
    assert cabin_region.known
    assert phone_role.role == "driver"
    assert events[0].event_type == "PHONE"


def test_event_evaluator_reports_false_missed_duplicate_and_delay():
    truth = [
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "1", "end_seconds": "2"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "8", "end_seconds": "9"},
    ]
    predictions = [
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "1.2", "end_seconds": "2.1"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "1.3", "end_seconds": "2.0"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "20", "end_seconds": "21"},
    ]
    report = evaluate(truth, predictions, video_minutes=1.0)
    metrics = report["event_types"]["PHONE"]
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["missed_events"] == 1
    assert report["duplicate_events"] == 1
    assert round(metrics["mean_time_to_detection_seconds"], 3) == 0.2


def test_association_evaluator_separates_context_role_and_pose_metrics():
    report = evaluate_associations(
        [
            {
                "vehicle_context_truth": "true",
                "vehicle_context_predicted": "true",
                "cabin_localized_truth": "true",
                "cabin_localized_predicted": "false",
                "occupant_role_truth": "driver",
                "occupant_role_predicted": "driver",
                "phone_to_driver_truth": "true",
                "phone_to_driver_predicted": "true",
                "pose_available": "true",
                "wrist_available": "false",
                "face_keypoints_available": "true",
            }
        ]
    )
    assert report["association_metrics"]["vehicle_context"]["accuracy"] == 1.0
    assert report["association_metrics"]["cabin_localization"]["accuracy"] == 0.0
    assert report["association_metrics"]["occupant_role"]["accuracy"] == 1.0
    assert report["pose_availability"]["wrist_rate"] == 0.0


def test_review_confirmation_requires_complete_integrity_checked_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(api_routes.settings, "evidence_dir", tmp_path)
    database = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(database)
    with Session(database) as session:
        user = User(id="reviewer", email="reviewer@example.com", password_hash="unused")
        video = Video(
            id="video",
            original_name="capture.mp4",
            storage_path="capture.mp4",
            sha256="0" * 64,
            size_bytes=1,
            mime_type="video/mp4",
            input_scope=InputScope.VEHICLE_CABIN_CROP,
        )
        job = AnalysisJob(id="job", video_id=video.id)
        event = Event(
            id="event",
            job_id=job.id,
            video_id=video.id,
            event_type=EventType.PHONE,
            confidence=0.8,
            frame_number=12,
            timestamp_seconds=0.4,
            track_id=3,
            occupant_role="driver",
            vehicle_context_id="vehicle:1",
            model_version="V2_TEST",
        )
        session.add_all([user, video, job, event])
        session.commit()

        with pytest.raises(HTTPException) as blocked:
            review_event(
                event.id,
                ReviewRequest(status=ReviewStatus.CONFIRMED),
                session,
                user,
            )
        assert blocked.value.status_code == 422
        assert session.get(Event, event.id).review_status == ReviewStatus.PENDING

        event_root = tmp_path / event.id
        event_root.mkdir()
        original_path = event_root / "original_keyframe.jpg"
        annotated_path = event_root / "annotated_keyframe.jpg"
        clip_path = event_root / "evidence.mp4"
        original_path.write_bytes(b"original")
        annotated_path.write_bytes(b"annotated")
        clip_path.write_bytes(b"temporal-clip")
        trace = {
            "evidence_status": "COMPLETE",
            "files": {
                "original_keyframe.jpg": sha256_file(original_path),
                "annotated_keyframe.jpg": sha256_file(annotated_path),
                "evidence.mp4": sha256_file(clip_path),
            },
        }
        (event_root / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
        actual_hashes = {
            "original_keyframe.jpg": sha256_file(original_path),
            "annotated_keyframe.jpg": sha256_file(annotated_path),
            "evidence.mp4": sha256_file(clip_path),
            "trace.json": sha256_file(event_root / "trace.json"),
        }
        session.add(
            Evidence(
                event_id=event.id,
                original_path=str(original_path),
                annotated_path=str(annotated_path),
                sha256=evidence_package_sha256(actual_hashes),
            )
        )
        session.commit()

        trace["files"]["annotated_keyframe.jpg"] = "f" * 64
        (event_root / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
        with pytest.raises(HTTPException, match="complete, integrity-checked"):
            review_event(
                event.id,
                ReviewRequest(status=ReviewStatus.CONFIRMED),
                session,
                user,
            )

        trace["files"]["annotated_keyframe.jpg"] = sha256_file(annotated_path)
        (event_root / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
        reviewed = review_event(
            event.id,
            ReviewRequest(status=ReviewStatus.CONFIRMED, notes="Visible phone in hand"),
            session,
            user,
        )
        assert reviewed.review_status == ReviewStatus.CONFIRMED
        with pytest.raises(HTTPException) as duplicate:
            review_event(
                event.id,
                ReviewRequest(status=ReviewStatus.CONFIRMED, notes="duplicate retry"),
                session,
                user,
            )
        assert duplicate.value.status_code == 409


def test_raw_traffic_context_skips_untracked_vehicle_regions():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.vehicle_interval = 1
    analyzer._vehicle_region_cache = []
    analyzer.vehicle_detector = SimpleNamespace(
        predict=lambda _: [
            VehicleRegion(
                "vehicle:untracked:car:0",
                0.9,
                (0, 0, 20, 20),
                None,
                2,
                "car",
            )
        ]
    )
    analyzer.cabin_localizer = SimpleNamespace(
        locate=lambda *_: pytest.fail("untracked vehicles must fail before cabin localization")
    )
    analyzer.camera_cabin_calibration = None
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    video = SimpleNamespace(id="v", input_scope=InputScope.TRAFFIC_SCENE_WITH_VEHICLE_ROIS)
    assert analyzer._frame_contexts(frame, video, 0) == []


def test_raw_traffic_context_preserves_vehicle_and_cabin_boxes():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.vehicle_interval = 1
    analyzer._vehicle_region_cache = []
    analyzer.cabin_signature_max_delta = 0.2
    analyzer._cabin_signatures = {}
    analyzer._cabin_epochs = defaultdict(int)
    analyzer.vehicle_detector = SimpleNamespace(
        predict=lambda _: [VehicleRegion("vehicle:4", 0.9, (2, 3, 22, 23), 4, 2, "car")]
    )
    analyzer.cabin_localizer = SimpleNamespace(
        locate=lambda *_: CabinRegion((2, 1, 18, 11), 0.95, "CAMERA_CALIBRATION", "KNOWN_CABIN")
    )
    analyzer.camera_cabin_calibration = None
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    video = SimpleNamespace(id="v", input_scope=InputScope.TRAFFIC_SCENE_WITH_VEHICLE_ROIS)
    context = analyzer._frame_contexts(frame, video, 0)[0]
    assert context.vehicle_bbox == (2.0, 3.0, 22.0, 23.0)
    assert context.cabin_bbox == (4.0, 4.0, 20.0, 14.0)
    assert context.vehicle_track_id == 4


def test_cabin_geometry_change_rotates_context_identity():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.vehicle_interval = 1
    analyzer._vehicle_region_cache = []
    analyzer.cabin_signature_max_delta = 0.1
    analyzer._cabin_signatures = {}
    analyzer._cabin_epochs = defaultdict(int)
    analyzer.vehicle_detector = SimpleNamespace(
        predict=lambda _: [VehicleRegion("vehicle:4", 0.9, (0, 0, 20, 20), 4, 2, "car")]
    )
    regions = iter(
        [
            CabinRegion((1, 1, 19, 12), 0.95, "CAMERA_CALIBRATION", "KNOWN_CABIN"),
            CabinRegion((5, 5, 15, 19), 0.95, "CAMERA_CALIBRATION", "KNOWN_CABIN"),
        ]
    )
    analyzer.cabin_localizer = SimpleNamespace(locate=lambda *_: next(regions))
    analyzer.camera_cabin_calibration = None
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    video = SimpleNamespace(id="v", input_scope=InputScope.TRAFFIC_SCENE_WITH_VEHICLE_ROIS)
    first = analyzer._frame_contexts(frame, video, 0)[0]
    second = analyzer._frame_contexts(frame, video, 1)[0]
    assert first.context_id != second.context_id


def test_production_runtime_refuses_untrained_unlocked_v2():
    detector = SafetyDetector(Path("models/active/best.pt"), Path("models/model_config_v2.yaml"))
    with pytest.raises(RuntimeError, match="production runtime validation failed"):
        detector.validate_runtime("production")


def test_seatbelt_classifier_interval_reuses_track_evidence():
    calls = []
    evidence = SeatbeltEvidence(0.1, 0.8, 0.1, "TEST")
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.classifier_interval = 3
    analyzer._seatbelt_cache = {}
    analyzer.seatbelt_classifier = SimpleNamespace(predict=lambda _: calls.append(True) or evidence)
    detection = NormalizedDetection(0, "occupant_upper_body", 0.9, (0, 0, 10, 10), 7)
    cabin = np.zeros((12, 12, 3), dtype=np.uint8)

    assert analyzer._seatbelt_evidence("vehicle:1", 0, detection, cabin) is evidence
    assert analyzer._seatbelt_evidence("vehicle:1", 1, detection, cabin) is evidence
    assert analyzer._seatbelt_evidence("vehicle:1", 3, detection, cabin) is evidence
    assert len(calls) == 2


def test_external_test_freeze_requires_human_review_and_disjoint_data(tmp_path):
    video_path = tmp_path / "external.mp4"
    annotation_path = tmp_path / "events.json"
    video_path.write_bytes(b"real-video-placeholder")
    annotation_path.write_text("[]", encoding="utf-8")
    item = {
        "sample_id": "external-1",
        "dataset_role": "EXTERNAL_TEST",
        "source_id": "external-source",
        "camera_id": "external-camera",
        "video_id": "external-video",
        "video_path": str(video_path),
        "sha256": sha256_file(video_path),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "conditions": ["daylight"],
        "human_review_status": "APPROVED",
        "reviewer_id": "reviewer-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-02T00:00:00Z",
    }
    manifest_path = tmp_path / "external.jsonl"
    development_path = tmp_path / "development.jsonl"
    policy_path = tmp_path / "policy.yaml"
    output_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    development_path.write_text(
        json.dumps(
            {
                "sha256": "f" * 64,
                "source_id": "development-source",
                "camera_id": "development-camera",
                "video_id": "development-video",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path.write_text(
        """dataset_role: EXTERNAL_TEST
required_fields:
  - sample_id
  - source_id
  - camera_id
  - video_id
  - video_path
  - sha256
  - annotation_path
  - annotation_sha256
  - conditions
  - human_review_status
  - reviewer_id
  - reviewer_type
  - reviewed_at
required_condition_coverage: [daylight]
disjoint_dimensions: [sha256, source_id, camera_id, video_id]
minimum_independent_groups: {source_id: 1, camera_id: 1, video_id: 1}
""",
        encoding="utf-8",
    )

    frozen = freeze_external_test(manifest_path, development_path, policy_path, output_path)
    assert frozen["status"] == "FROZEN_EXTERNAL_TEST"
    assert frozen["human_review_status"] == "ALL_APPROVED"

    item["human_review_status"] = "PENDING"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="human review is not APPROVED"):
        freeze_external_test(manifest_path, development_path, policy_path, output_path)

    item["human_review_status"] = "APPROVED"
    item["reviewer_type"] = "AI"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer_type must be HUMAN"):
        freeze_external_test(manifest_path, development_path, policy_path, output_path)

    item["reviewer_type"] = "HUMAN"
    item["camera_id"] = "development-camera"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="camera_id overlaps development data"):
        freeze_external_test(manifest_path, development_path, policy_path, output_path)

    item["camera_id"] = "external-camera"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_external_test(manifest_path, development_path, policy_path, output_path)
