"""Scientific safety regressions; generated tensors are fixtures, never measured metrics."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from training.dmd import action_adapter as module
from training.dmd.action_checkpoint import (
    COARSE_CLASSES,
    FEATURE_CONTRACT,
    CheckpointSafetyError,
    sha256_file,
    validate_checkpoint_manifest,
)


def test_wrong_loso_fallback_is_rejected(tmp_path, monkeypatch):
    other = tmp_path / "candidate/v6_action_ablation_001/hierarchical_loss_only_loso_gZ-36.pt"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"must never deserialize another fold")
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("wrong fold deserialized"))
    adapter = module.ActionInferenceAdapter(heldout_subject="gC-14")
    assert adapter._deep_model is None
    scores = adapter.predict(np.zeros((20, 20, 3), dtype=np.uint8), None)
    assert scores.provenance["actual_mode"] == "RULE_ONLY"
    assert scores.provenance["fallback_reason"] == "MISSING_COMPATIBLE_LOSO_CHECKPOINT"


def test_missing_checkpoint_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path)
    with pytest.raises(CheckpointSafetyError, match="MISSING_COMPATIBLE"):
        module.ActionInferenceAdapter(missing_checkpoint_policy="ERROR")


@pytest.fixture
def contract(tmp_path):
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"fixture")
    source = tmp_path / "training_manifest.json"
    source.write_text("{}")
    manifest = {
        "training_subject_ids": ["gB-9", "gZ-36"],
        "validation_subject_ids": [],
        "intended_heldout_subject": "gC-14",
        "architecture": "HierarchicalResNetGRUClassifier",
        "feature_preprocessing": deepcopy(FEATURE_CONTRACT),
        "class_mapping": COARSE_CLASSES,
        "sha256": sha256_file(checkpoint),
        "backbone_sha256": "a" * 64,
        "sample_period_seconds": 0.5,
        "training_dataset_provenance": [{"path": str(source), "sha256": sha256_file(source)}],
    }
    return checkpoint, tmp_path / "head.manifest.json", manifest


def validate(contract):
    checkpoint, path, manifest = contract
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return validate_checkpoint_manifest(
        checkpoint, path, "gC-14", "HierarchicalResNetGRUClassifier"
    )


def test_checkpoint_manifest_matches_subject_split(contract):
    assert validate(contract)["intended_heldout_subject"] == "gC-14"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("sha256", "0" * 64, "CHECKPOINT_SHA_MISMATCH"),
        ("training_subject_ids", ["gC-14"], "OVERLAP"),
        ("validation_subject_ids", ["gZ-36"], "OVERLAP"),
        ("intended_heldout_subject", "gZ-36", "WRONG_LOSO"),
        ("architecture", "flat", "ARCHITECTURE"),
        ("class_mapping", list(reversed(COARSE_CLASSES)), "CLASS_MAPPING"),
        ("feature_preprocessing", {"embedding_dim": 256}, "PREPROCESSING"),
        ("training_subject_ids", ["gA-1"], "PROTECTED"),
    ],
)
def test_checkpoint_contract_rejects_incompatibilities(contract, field, value, reason):
    contract[2][field] = value
    with pytest.raises(CheckpointSafetyError, match=reason):
        validate(contract)


def test_fold_selection_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path)
    adapters = [module.ActionInferenceAdapter() for _ in range(2)]
    assert adapters[0].checkpoint_path == adapters[1].checkpoint_path


def test_hierarchy_output_semantics():
    from training.dmd.v6_action_hierarchy import coarse_targets, hierarchical_probabilities

    assert coarse_targets(torch.arange(6)).tolist() == [0, 1, 1, 1, 1, 2]
    coarse = torch.tensor([[0.4, 1.2, -0.2]])
    fine = torch.tensor([[0.1, -0.2, 0.4, 0.7]])
    flat = hierarchical_probabilities(coarse, fine)
    reconstructed = torch.stack([flat[:, 0], flat[:, 1:5].sum(1), flat[:, 5]], dim=1)
    torch.testing.assert_close(reconstructed, torch.softmax(coarse, dim=1))


def test_sequence_state_isolation_and_no_padding():
    adapter = module.ActionInferenceAdapter(mode="RULE_ONLY")
    key = ("gC-14", "session1", "BODY", "track1")
    assert adapter.observe_embedding(np.ones(512), key, 0.0) is None
    for i in range(1, 9):
        sequence = adapter.observe_embedding(np.ones(512) * i, key, i * 0.5)
    assert sequence.shape == (9, 512)
    for index in range(4):
        switched = list(key)
        switched[index] += "_changed"
        assert adapter.observe_embedding(np.ones(512), tuple(switched), 5.0) is None
        assert len(adapter.embedding_buffer) == 1


def test_missing_frame_resets_history_and_wrong_dimension_rejected():
    adapter = module.ActionInferenceAdapter(mode="RULE_ONLY")
    key = ("gC-14", "s2", "BODY", "1")
    adapter.observe_embedding(np.ones(512), key, 0.0)
    adapter.observe_embedding(np.ones(512), key, 10.0)
    assert len(adapter.embedding_buffer) == 1
    with pytest.raises(ValueError, match="512"):
        adapter.observe_embedding(np.ones(256), key, 10.5)


def test_confusion_matrix_reconciles_with_raw_labels():
    from training.dmd.v22_evaluation import metrics_from_raw

    measured = metrics_from_raw(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), 3)
    assert measured["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert measured["support"] == [2, 1, 1]
    assert measured["sample_count"] == 4


def test_macro_f1_matches_confusion_matrix():
    from training.dmd.v22_evaluation import confusion_metrics, verify_report_metrics

    metrics = confusion_metrics([[1687, 42, 121], [58, 839, 83], [102, 79, 589]])
    assert metrics["macro_f1"] == pytest.approx(0.8437530466860692, abs=1e-12)
    assert sum(metrics["support"]) == metrics["sample_count"] == 3600
    with pytest.raises(ValueError, match="INCONSISTENT_REPORT"):
        verify_report_metrics(metrics, {"macro_f1": 0.8516})
    with pytest.raises(ValueError, match="INCONSISTENT_REPORT"):
        verify_report_metrics(metrics, {"sample_count": 3599})


def test_per_subject_metrics_aggregate_correctly():
    from training.dmd.v22_evaluation import metrics_from_raw

    labels = [np.array([0, 1, 1]), np.array([2, 0, 2, 2])]
    predictions = [np.array([0, 0, 1]), np.array([2, 1, 1, 2])]
    folds = [metrics_from_raw(y, p, 3) for y, p in zip(labels, predictions, strict=True)]
    pooled = metrics_from_raw(np.concatenate(labels), np.concatenate(predictions), 3)
    assert np.array_equal(
        np.sum([m["confusion_matrix"] for m in folds], axis=0), pooled["confusion_matrix"]
    )


def coverage_fixture():
    return [
        {
            "subject_id": "gC-14",
            "source_frames": 100,
            "sample_stride": 15,
            "frame_indices": list(range(0, 100, 15)),
            "fps": 10,
            "evaluated_seconds": 10,
            "gt_count": 2,
            "scope": "FULL_COHORT_AUTONOMOUS_EXPERIMENT",
        }
    ]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("frame_indices", [0, 15], "FRAME_COVERAGE"),
        ("scope", "SHORT_SMOKE_TEST", "SMOKE_CANNOT"),
        ("evaluated_seconds", 60, "EXPOSURE"),
        ("gt_count", 1, "GT_COUNT"),
    ],
)
def test_full_cohort_cannot_silently_shrink(field, value, reason):
    from training.dmd.v22_evaluation import validate_full_coverage

    sessions = coverage_fixture()
    sessions[0][field] = value
    with pytest.raises(ValueError, match=reason):
        validate_full_coverage(sessions, ["gC-14"], 2)


def test_full_cohort_manifest_cannot_skip_subject():
    from training.dmd.v22_evaluation import validate_full_coverage

    with pytest.raises(ValueError, match="INCOMPLETE_COHORT"):
        validate_full_coverage(coverage_fixture(), ["gC-14", "gZ-36"], 20)


def test_gt_cannot_enter_stage1_inference():
    import inspect

    from training.dmd.run_autonomous_v20 import AutonomousVideoInferenceEngine
    from training.dmd.run_verified_v22 import extract_session, replay_verified_record

    for function in (
        AutonomousVideoInferenceEngine.run_session_inference,
        extract_session,
        replay_verified_record,
    ):
        signature = inspect.signature(function)
        assert not {"annotation_path", "gt", "labels", "matching_policy"} & set(
            signature.parameters
        )
        source = inspect.getsource(function)
        assert "parse_dmd_openlabel" not in source


def test_frozen_matcher_hash_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert sha256_file(root / "training/dmd/temporal_eval.py") == (
        "438f6e321df37f045aa89fc8972828f9e6bd2fd1e600f89f707b76fcc37f8db6"
    )


def test_driver_role_without_seat_calibration_is_unknown():
    from training.dmd.run_verified_v22 import replay_verified_record

    record = {
        "poses": [],
        "candidates": [{"local_box": [1, 1, 5, 5], "confidence": 0.9}],
        "observation": {"timestamp": 0.0},
    }
    observation, detail = replay_verified_record(record, None)
    assert observation.occupant_role == "UNKNOWN"
    assert observation.confidence == 0
    assert detail["candidates"][0]["association"]["state"] == "UNKNOWN"


def test_boundary_metrics_do_not_inflate_short_alert():
    from training.dmd.v22_evaluation import boundary_metrics

    gt = [{"start_seconds": 1.0, "end_seconds": 10.0}]
    predictions = [{"start_seconds": 1.0, "end_seconds": 1.5}]
    boundary = boundary_metrics([], gt, predictions)
    assert boundary["event_coverage"] == pytest.approx([0.5 / 9.0])
    assert boundary["predicted_durations_seconds"] == [0.5]


@pytest.mark.parametrize("limit,expected_frames,expected_samples", [(None, 10, 4), (1, 1, 1)])
def test_actual_decode_exposure_and_complete_persistence(
    monkeypatch, limit, expected_frames, expected_samples
):
    from types import SimpleNamespace

    import cv2
    from training.dmd import run_autonomous_v20 as baseline

    class Capture:
        position = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            return 10 if property_id == cv2.CAP_PROP_FRAME_COUNT else 2

        def read(self):
            if self.position >= 10:
                return False, None
            self.position += 1
            return True, np.zeros((20, 20, 3), dtype=np.uint8)

        def grab(self):
            if self.position >= 10:
                return False
            self.position += 1
            return True

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    engine = baseline.AutonomousVideoInferenceEngine(sample_stride=3)
    empty_detector = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=[])])
    monkeypatch.setattr(engine, "_get_specialist", lambda: empty_detector)
    monkeypatch.setattr(engine, "_get_auxiliary", lambda: empty_detector)
    records = []
    measured = engine.run_session_inference(Path("fixture.mp4"), {}, limit, records.append)
    assert measured["total_frames"] == expected_frames
    assert measured["total_seconds"] == expected_frames / 2
    assert measured["processed_frames"] == expected_samples == len(records)
    assert [row["frame_idx"] for row in records] == list(range(0, expected_frames, 3))
    assert measured["wall_time_seconds"] > 0


def test_active_boundary_reporter_tracks_evidence_and_flushes_without_inventing_tail():
    from backend.ai.events import Observation
    from training.dmd.v22_boundary_diagnostic import ActiveBoundaryReporter

    reporter = ActiveBoundaryReporter({"min_positive_seconds": 0.5, "min_observations": 2})
    for timestamp in (0.0, 0.5, 1.0, 1.5):
        reporter.add(
            Observation(
                timestamp=timestamp,
                class_name="phone",
                confidence=0.95,
                track_id=1,
                occupant_role="driver",
                vehicle_context_id="fixture",
                phone_context="HANDHELD_USE",
            )
        )
    events = reporter.finish(3.0)
    assert len(events) == 1
    assert events[0]["start_seconds"] == 0.0
    assert events[0]["end_seconds"] == 1.5
    assert events[0]["emitted_at_seconds"] == 0.5
    assert reporter.active == {}
