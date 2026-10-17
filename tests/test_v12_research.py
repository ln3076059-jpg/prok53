"""Regression tests for V12 uncertainty, causality, rights, and portable inference."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from training.dmd.behavior_roi import BehaviorRoiExtractor, Box, RoiStrategy
from training.dmd.causal_motion import (
    CausalMotionEncoder,
    causal_running_standardize,
    spatial_residual_standardize,
)
from training.dmd.driver_phone_association import DriverPhoneAssociation, OccupantTrack
from training.dmd.high_res_roi import HighResolutionRoiPipeline, frame_sha256
from training.dmd.online_candidate import NumericHead, OnlineConfig, OnlineE3, ResearchCandidate
from training.dmd.research_permissions import FIT_OPERATIONS, require_operation, require_subject
from training.dmd.v12_diagnostics import classification_metrics, event_counts
from training.dmd.v12_reports import read_native_log


def frames(count=24):
    return np.random.default_rng(85).integers(0, 256, (count, 64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_native_log_reader_supports_powershell_encodings(tmp_path, encoding):
    path = tmp_path / "build.log"
    path.write_text("built in 1.0s", encoding=encoding)
    assert read_native_log(path) == "built in 1.0s"


def test_online_e3_matches_batch_and_is_future_invariant():
    data = frames()
    raw = CausalMotionEncoder().encode_sequence(data)
    expected = causal_running_standardize(spatial_residual_standardize(raw))
    stream = OnlineE3()
    actual = [
        stream.push(frame, i * 15 / 29.76, ("s", "BODY", "d")).feature
        for i, frame in enumerate(data)
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    data[18:] = 0
    another = OnlineE3()
    modified = [
        another.push(frame, i * 15 / 29.76, ("s", "BODY", "d")).feature
        for i, frame in enumerate(data)
    ]
    np.testing.assert_array_equal(actual[:18], modified[:18])


@pytest.mark.parametrize("key", [("new", "BODY", "d"), ("s", "FACE", "d"), ("s", "BODY", "new")])
def test_online_stream_identity_reset_and_cold_start(key):
    stream = OnlineE3()
    for i, frame in enumerate(frames()):
        result = stream.push(frame, i * 15 / 29.76, ("s", "BODY", "d"))
        assert result.ready == (i >= 8)
    result = stream.push(frames(1)[0], 0, key)
    assert not result.ready and result.history_count == 0
    assert "STREAM_RESET" in result.reasons
    assert not result.feature.any()


def test_gap_missing_view_and_out_of_order_handling():
    stream = OnlineE3()
    frame = frames(1)[0]
    stream.push(frame, 0, ("s", "BODY", "d"))
    result = stream.push(frame, 5, ("s", "BODY", "d"))
    assert "CADENCE_GAP_RESET" in result.reasons
    with pytest.raises(ValueError, match="increase"):
        stream.push(frame, 4, ("s", "BODY", "d"))
    assert stream.timestamp == 5
    missing = stream.push(None, 5 + 15 / 29.76, ("s", "BODY", "d"))
    assert missing.feature is None and "MISSING_VIEW" in missing.reasons
    assert len(stream.frames) == len(stream.history) == 0


def test_stream_state_bounded_and_zero_variance_finite():
    stream = OnlineE3(OnlineConfig(history=10))
    for i in range(150):
        result = stream.push(np.zeros((64, 64, 3), np.uint8), i * 15 / 29.76, ("s", "BODY", "d"))
    assert len(stream.frames) == 8 and len(stream.history) == 10
    assert np.isfinite(result.feature).all()


@pytest.mark.parametrize("operation", sorted(FIT_OPERATIONS | {"upload"}))
def test_permission_gate_reloads_record_and_never_infers_train_from_diagnostics(
    tmp_path, operation
):
    record = tmp_path / "rights.md"
    record.write_text("RIGHTS_ACCEPTED = false\nTEMPORAL_DEVELOPMENT_ELIGIBLE = true\n")
    require_operation("diagnostics", record)
    with pytest.raises(PermissionError, match="RIGHTS_ACCEPTED"):
        require_operation(operation, record)
    record.write_text(
        "RIGHTS_ACCEPTED = true\nTRAINING_ALLOWED = true\n"
        "EXTERNAL_UPLOAD_ALLOWED = true\nLABELS_APPROVED = true\n"
        "GROUP_SPLIT_VERIFIED = true\n"
    )
    require_operation(operation, record)
    record.write_text("RIGHTS_ACCEPTED = true\n")
    with pytest.raises(PermissionError, match="explicit"):
        require_operation(operation, record)


def test_conflicting_rights_fail_closed(tmp_path):
    record = tmp_path / "rights.md"
    record.write_text("RIGHTS_ACCEPTED = false\nRIGHTS_ACCEPTED = true\n")
    with pytest.raises(PermissionError, match="Conflicting"):
        require_operation("linear_head", record)


@pytest.mark.parametrize("subject", ["gZ-37", "gE-28", "unknown", "../gE-28"])
def test_subject_guard_before_io(subject):
    with pytest.raises(PermissionError):
        require_subject(subject)


def test_v11_fitting_entrypoints_block_before_loading_data(monkeypatch, tmp_path):
    import training.dmd.v11_generalization_experiments as gen
    import training.dmd.v11_representation_experiments as rep

    def denied(*args):
        raise PermissionError("training denied")

    def unexpected(*args):
        pytest.fail("data loaded before permission check")

    for module in (gen, rep):
        monkeypatch.setattr(module, "require_operation", denied)
        monkeypatch.setattr(module, "_load_subject", unexpected)
        with pytest.raises(PermissionError):
            module.run(tmp_path)


def test_mounted_context_requires_independent_verification():
    args = dict(
        phone_box=Box(5, 5, 15, 15),
        occupants=(),
        phone_context="MOUNTED_OR_STATIC",
        phone_confidence=0.9,
    )
    assert DriverPhoneAssociation().associate(**args).state == "UNKNOWN"
    result = DriverPhoneAssociation().associate(**args, context_verified=True)
    assert result.state == "MOUNTED_OR_STATIC"


def test_overlapping_occupants_abstain_independent_of_order():
    driver = OccupantTrack("d", "driver", Box(0, 0, 100, 100), True, 0.99)
    passenger = OccupantTrack("p", "front_passenger", Box(20, 0, 100, 100), True, 0.8)
    for tracks in ((driver, passenger), (passenger, driver)):
        result = DriverPhoneAssociation().associate(
            Box(25, 20, 40, 40), tracks, phone_context="HANDHELD_USE", phone_confidence=0.9
        )
        assert result.state == "UNKNOWN"
        assert result.reason_codes == ("AMBIGUOUS_OCCUPANT_CONTAINMENT",)


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError, match="finite"):
        DriverPhoneAssociation().associate(
            Box(0, 0, 1, 1), (), phone_context="UNKNOWN", phone_confidence=float("nan")
        )


def test_dynamic_roi_uses_actual_pixels_beyond_driver_boundary():
    image = np.zeros((100, 120, 3), np.uint8)
    image[30:40, 85:95] = 255
    crop = BehaviorRoiExtractor().extract(
        image,
        Box(0, 0, 80, 100),
        RoiStrategy.DYNAMIC_PADDED,
        source_camera="BODY",
        phone_box=Box(85, 30, 95, 40),
    )[0]
    assert crop.image.max() == 255
    assert crop.provenance.crop_box[2] > 95
    assert crop.provenance.padding_ltrb == (0, 0, 0, 0)


def test_high_resolution_roi_preserves_pixels_hash_and_boundary_provenance():
    image = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
    phone = Box(112, 65, 125, 82)
    crop = HighResolutionRoiPipeline(padding=0.2).extract(
        image,
        Box(20, 10, 118, 78),
        RoiStrategy.DYNAMIC_PADDED,
        source_camera="BODY",
        phone_box=phone,
    )[0]
    assert crop.frame_sha256 == frame_sha256(image)
    assert crop.output_dimensions != (64, 64)
    assert crop.provenance["scale_xy"] == (1.0, 1.0)
    assert crop.edge_clipped
    assert 0 < crop.phone_coverage < 1


def test_numeric_export_reload_matches_sklearn_synthetic_only(tmp_path):
    rng = np.random.default_rng(85)
    x = rng.normal(size=(90, 6))
    y = np.tile(np.arange(3), 30)
    scaler = StandardScaler().fit(x)
    classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear")).fit(
        scaler.transform(x), y
    )
    head = NumericHead.from_fitted(
        scaler,
        classifier,
        {
            "dataset_hash": "SYNTHETIC_ONLY",
            "split_hash": "SYNTHETIC_ONLY",
            "feature_version": "fixture",
            "seed": 85,
            "training_subjects": ["synthetic"],
        },
    )
    path = tmp_path / "head.json"
    head.save(path)
    restored = NumericHead.load(path)
    pred, probs = restored.predict(x)
    np.testing.assert_array_equal(pred, classifier.predict(scaler.transform(x)))
    np.testing.assert_allclose(probs, classifier.predict_proba(scaler.transform(x)), atol=1e-12)
    assert restored.sha256 == head.sha256
    payload = json.loads(path.read_text())
    payload["classes"] = payload["classes"][::-1]
    with pytest.raises(ValueError, match="class mapping"):
        NumericHead(payload)


def synthetic_head():
    return NumericHead(
        {
            "schema": 1,
            "classes": ["SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY"],
            "mean": [0.0] * 444,
            "scale": [1.0] * 444,
            "weights": [[0.0] * 444] * 3,
            "bias": [-4.0, 4.0, -4.0],
            "provenance": {
                "dataset_hash": "SYNTHETIC",
                "split_hash": "SYNTHETIC",
                "feature_version": "E3",
                "seed": 85,
                "training_subjects": ["synthetic"],
            },
        }
    )


def test_candidate_opt_in_passenger_and_unknown_abstention():
    track = OccupantTrack("p", "front_passenger", Box(0, 0, 100, 100), True, 1.0)
    association = DriverPhoneAssociation().associate(
        Box(5, 5, 10, 10), (track,), phone_context="HANDHELD_USE", phone_confidence=0.95
    )
    candidate = ResearchCandidate(synthetic_head(), {})
    with pytest.raises(RuntimeError, match="disabled"):
        candidate.push(frames(1)[0], 0, ("s", "BODY", "p"), association, 0.95, "HANDHELD_USE")
    candidate.enabled = True
    for i, frame in enumerate(frames()):
        row = candidate.push(
            frame, i * 15 / 29.76, ("s", "BODY", "p"), association, 0.95, "HANDHELD_USE"
        )
        assert row["events"] == [] and row["abstained"]


def test_candidate_events_have_emission_time_and_stable_id():
    track = OccupantTrack("d", "driver", Box(0, 0, 100, 100), True, 1.0)
    association = DriverPhoneAssociation().associate(
        Box(5, 5, 10, 10), (track,), phone_context="HANDHELD_USE", phone_confidence=0.95
    )
    all_ids = []
    for _ in range(2):
        candidate = ResearchCandidate(synthetic_head(), {}, enabled=True)
        events = []
        for i, frame in enumerate(frames()):
            row = candidate.push(
                frame, i * 15 / 29.76, ("s", "BODY", "d"), association, 0.95, "HANDHELD_USE"
            )
            events.extend(row["events"])
        assert events
        assert all(e["emission_time"] >= e["start_timestamp"] for e in events)
        all_ids.append([e["event_id"] for e in events])
    assert all_ids[0] == all_ids[1]


def test_metric_recompute_includes_missing_class_and_safe_collapse():
    result = classification_metrics(np.array([0, 1, 2]), np.array([1, 1, 1]))
    assert result["safe_recall"] == 0
    assert result["confusion_matrix"] == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert result["macro_f1"] == 0.1667


def test_duplicate_predictions_count_as_false_positives():
    matches = [{"ground_truth": {"event_id": "same"}}] * 2
    assert event_counts(matches, [], [{"event_id": "miss"}]) == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "duplicate_predictions": 1,
    }
