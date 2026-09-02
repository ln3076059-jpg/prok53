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

from backend.ai.detector import SafetyDetector
from backend.ai.evidence import EventEvidenceBuffer
from backend.api import routes as api_routes
from backend.core.config import DEVELOPMENT_SECRET_KEY, Settings
from backend.models.entities import AnalysisJob
from backend.services import analysis_worker
from backend.services.analysis_worker import InferenceWorker
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


def test_production_settings_reject_public_development_secret():
    with pytest.raises(ValueError, match="public development SECRET_KEY"):
        Settings(
            _env_file=None,
            app_env="production",
            secret_key=DEVELOPMENT_SECRET_KEY,
        )


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
