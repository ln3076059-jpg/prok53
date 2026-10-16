from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.ai.detector import SafetyDetector
from backend.ai.evidence import EventEvidenceBuffer
from backend.api import routes as api_routes
from backend.api.dependencies import export_user, reviewer_user
from backend.core.config import DEVELOPMENT_SECRET_KEY, Settings
from backend.database import Base
from backend.models.entities import (
    AnalysisJob,
    Event,
    EventType,
    JobStatus,
    Review,
    ReviewStatus,
    User,
    Video,
)
from backend.schemas import ReviewRequest
from backend.services import analysis_worker
from backend.services.analysis_queue import DurableAnalysisQueue
from backend.services.analysis_worker import InferenceWorker
from backend.services.uploads import validate_video_container
from backend.services.video_source import OpenCVVideoSource, RTSPSource, WebcamSource
from training.common import sha256_file
from training.lock_model import validate_calibration_binding


def test_inference_worker_serializes_concurrent_jobs(monkeypatch, tmp_path):
    jobs = {
        "job-1": SimpleNamespace(video_id="video-1"),
        "job-2": SimpleNamespace(video_id="video-2"),
    }
    videos = {
        "video-1": SimpleNamespace(id="video-1"),
        "video-2": SimpleNamespace(id="video-2"),
    }

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, model, identity):
            return jobs.get(identity) if model is AnalysisJob else videos.get(identity)

        def rollback(self):
            pass

        def commit(self):
            pass

    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0}

    class FakeAnalyzer:
        def __init__(self, *_):
            pass

        def analyze(self, *_):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(analysis_worker, "VideoAnalyzer", FakeAnalyzer)
    worker = InferenceWorker(FakeSession, SimpleNamespace(), tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker.run, jobs))

    assert state["maximum"] == 1


def test_video_source_releases_failed_capture(monkeypatch):
    capture = SimpleNamespace(
        isOpened=lambda: False,
        release=lambda: setattr(capture, "released", True),
    )
    capture.released = False
    monkeypatch.setattr("backend.services.video_source.cv2.VideoCapture", lambda _: capture)

    with pytest.raises(ValueError, match="cannot be opened"):
        OpenCVVideoSource("missing.mp4")

    assert capture.released is True


@pytest.mark.parametrize("source", ["http://camera/live", "rtsp://", "not-a-url"])
def test_rtsp_source_rejects_malformed_url_before_open(source):
    with pytest.raises(ValueError, match="RTSP source"):
        RTSPSource(source)


@pytest.mark.parametrize("source", [-1, "0", True])
def test_webcam_source_rejects_invalid_device_index(source):
    with pytest.raises(ValueError, match="non-negative integer"):
        WebcamSource(source)


def test_evidence_paths_cannot_escape_configured_root(tmp_path, monkeypatch):
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(api_routes.settings, "evidence_dir", evidence_root)
    with pytest.raises(HTTPException) as exc:
        api_routes._event_evidence_root("../outside")
    assert exc.value.status_code == 400

    buffer = EventEvidenceBuffer(1.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="escapes the evidence root"):
        buffer.register("../outside", evidence_root, 0.0, frame, frame, {})


@pytest.mark.parametrize("value", ["=cmd", "+formula", "-formula", "@formula", "\tformula"])
def test_csv_export_escapes_spreadsheet_formulas(value):
    assert api_routes._csv_safe(value).startswith("'")


def test_review_and_export_dependencies_enforce_roles():
    reviewer = User(email="reviewer@example.test", password_hash="x", role="reviewer")
    auditor = User(email="auditor@example.test", password_hash="x", role="auditor")
    viewer = User(email="viewer@example.test", password_hash="x", role="viewer")

    assert reviewer_user(user=reviewer) is reviewer
    assert export_user(user=auditor) is auditor
    with pytest.raises(HTTPException) as exc:
        reviewer_user(user=auditor)
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        export_user(user=viewer)
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    ("suffix", "header"),
    [
        (".mp4", b"\x00\x00\x00\x18ftypisom"),
        (".avi", b"RIFF\x10\x00\x00\x00AVI "),
        (".webm", b"\x1a\x45\xdf\xa3\x00\x00\x00\x00"),
    ],
)
def test_video_container_validation_accepts_matching_signatures(tmp_path, suffix, header):
    path = tmp_path / f"video{suffix}"
    path.write_bytes(header)
    validate_video_container(path)


def test_video_container_validation_does_not_read_entire_upload(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"payload")

    def fail_full_read(_):
        raise AssertionError("validate_video_container must not call Path.read_bytes")

    monkeypatch.setattr(type(path), "read_bytes", fail_full_read)
    validate_video_container(path)


def test_video_container_validation_rejects_spoofed_and_empty_files(tmp_path):
    spoofed = tmp_path / "not-a-video.mp4"
    spoofed.write_bytes(b"this is plain text")
    with pytest.raises(HTTPException) as exc:
        validate_video_container(spoofed)
    assert exc.value.status_code == 415

    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(HTTPException) as exc:
        validate_video_container(empty)
    assert exc.value.status_code == 400


def test_review_transition_rejects_stale_concurrent_writer(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'review.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as seed:
        user = User(id="reviewer", email="reviewer@example.test", password_hash="x")
        video = Video(
            id="video",
            original_name="video.mp4",
            storage_path="video.mp4",
            sha256="a" * 64,
            size_bytes=12,
            mime_type="video/mp4",
        )
        job = AnalysisJob(id="job", video_id="video", status=JobStatus.COMPLETED)
        event = Event(
            id="event",
            job_id="job",
            video_id="video",
            event_type=EventType.PHONE,
            confidence=0.8,
            frame_number=1,
            timestamp_seconds=1.0,
            occupant_role="driver",
            vehicle_context_id="vehicle",
            model_version="test",
        )
        seed.add_all([user, video, job, event])
        seed.commit()

    first = session_factory()
    second = session_factory()
    try:
        first_event = first.get(Event, "event")
        second_event = second.get(Event, "event")
        first_user = first.get(User, "reviewer")
        second_user = second.get(User, "reviewer")
        api_routes._commit_review_transition(
            first,
            first_event,
            first_user,
            ReviewRequest(status=ReviewStatus.NEEDS_REVIEW, notes="ambiguous"),
        )
        with pytest.raises(HTTPException) as exc:
            api_routes._commit_review_transition(
                second,
                second_event,
                second_user,
                ReviewRequest(status=ReviewStatus.REJECTED, notes="stale"),
            )
        assert exc.value.status_code == 409
        with session_factory() as check:
            assert check.get(Event, "event").review_status == ReviewStatus.NEEDS_REVIEW
            assert len(list(check.query(Review))) == 1
    finally:
        first.close()
        second.close()


def test_production_settings_reject_public_development_secret():
    with pytest.raises(ValueError, match="public development SECRET_KEY"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key=DEVELOPMENT_SECRET_KEY,
        )


def test_production_settings_require_durable_queue():
    with pytest.raises(ValueError, match="ANALYSIS_QUEUE_BACKEND=database"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="x" * 32,
            analysis_queue_backend="background",
        )


def test_production_settings_reject_placeholder_secrets_and_database_passwords():
    with pytest.raises(ValueError, match="placeholder SECRET_KEY"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="replace-with-a-random-secret-of-at-least-32-characters",
            database_url="mysql+pymysql://roadwatch:strong@mysql/roadwatch",
            analysis_queue_backend="database",
        )
    with pytest.raises(ValueError, match="placeholder database credentials"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="x" * 32,
            database_url="mysql+pymysql://roadwatch:replace-me@mysql/roadwatch",
            analysis_queue_backend="database",
        )


def test_durable_queue_claims_each_persisted_job_once(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'queue.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        session.add(
            Video(
                id="video",
                original_name="video.mp4",
                storage_path="video.mp4",
                sha256="b" * 64,
                size_bytes=12,
                mime_type="video/mp4",
            )
        )
        session.add_all(
            [AnalysisJob(id="job-1", video_id="video"), AnalysisJob(id="job-2", video_id="video")]
        )
        session.commit()

    queue = DurableAnalysisQueue(session_factory)
    queue.enqueue("job-1")
    claimed = {queue.claim_next(), queue.claim_next()}
    assert claimed == {"job-1", "job-2"}
    assert queue.claim_next() is None
    with session_factory() as session:
        assert {session.get(AnalysisJob, job_id).status for job_id in claimed} == {
            JobStatus.RUNNING
        }


def test_calibration_binding_rejects_model_and_threshold_mismatch():
    payload = {
        "status": "CALIBRATED_ON_VALIDATION",
        "created_at": "2026-09-02T00:00:00+00:00",
        "method": "TEST",
        "model": {"sha256": "a" * 64},
        "validation_dataset_manifest": {"sha256": "b" * 64},
        "score_artifact": {"sha256": "c" * 64},
        "thresholds": {"phone": 0.4},
        "classes": {"phone": {}},
        "test_rows_used": 0,
    }
    with pytest.raises(ValueError, match="does not match locked model weights"):
        validate_calibration_binding(payload, {"phone": "d" * 64}, {"phone": 0.4})

    payload["model"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="runtime threshold does not match"):
        validate_calibration_binding(payload, {"phone": "d" * 64}, {"phone": 0.5})


def test_production_runtime_rejects_wrong_locked_weight_hash(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"model")
    classifier = tmp_path / "seatbelt-classifier.pt"
    classifier.write_bytes(b"classifier")
    lock_path = tmp_path / "model-lock.json"
    config_path = tmp_path / "config.yaml"
    config = {
        "experiment_id": "V2_TEST",
        "model_version": "V2_TEST",
        "activation_state": "PRODUCTION_APPROVED",
        "weights": str(weights),
        "class_names": {
            0: "phone",
            1: "seatbelt_fastened",
            2: "seatbelt_unfastened",
        },
        "thresholds": {
            "phone": 0.4,
            "seatbelt_fastened": 0.5,
            "seatbelt_unfastened": 0.5,
        },
        "threshold_status": "CALIBRATED_ON_VALIDATION",
        "specialized_detectors": {"enabled": False},
        "auxiliary_models": {"seatbelt_classifier_weights": str(classifier)},
        "vehicle_context": {},
        "temporal": {},
        "activation_policy": {
            "require_locked_model_record": True,
            "model_lock_path": str(lock_path),
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    detector = SafetyDetector(weights, config_path)
    lock_path.write_text(
        json.dumps(
            {
                "production_approved": True,
                "activation_state": "ACTIVE",
                "config_sha256": detector.config_sha256,
                "experiment_id": "V2_TEST",
                "weights_sha256": sha256_file(weights),
                "component_weight_sha256": {"seatbelt_classifier": "f" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="component weight hashes"):
        detector.validate_runtime("production")
