import json
from pathlib import Path

import numpy as np
import pytest

from training.dmd.behavior_roi import (
    BehaviorRoiExtractor,
    Box,
    EvidenceState,
    RoiStrategy,
)
from training.dmd.causal_motion import (
    CausalMotionConfig,
    CausalMotionEncoder,
    causal_running_standardize,
    spatial_residual_standardize,
)
from training.dmd.driver_phone_association import (
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.v11_representation_experiments import _load_subject, _metrics

ROOT = Path(__file__).resolve().parents[1]


def test_causal_motion_never_reads_future_frames():
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    frames[0] = 10
    frames[1] = 30
    encoder = CausalMotionEncoder(CausalMotionConfig(lags=(1, 2), map_size=(4, 4)))
    before = encoder.encode_at(frames, 2)
    frames[3:] = 255
    after = encoder.encode_at(frames, 2)
    np.testing.assert_array_equal(before, after)


def test_causal_motion_initial_history_is_zero_and_dimensions_are_stable():
    frames = np.full((2, 8, 8, 3), 50, dtype=np.uint8)
    encoder = CausalMotionEncoder(CausalMotionConfig(lags=(1, 3), map_size=(3, 4)))
    result = encoder.encode_sequence(frames)
    assert result.shape == (2, encoder.feature_dimensions)
    assert not result[0].any()


def test_running_standardization_is_causal_and_future_invariant():
    features = np.arange(48, dtype=np.float32).reshape(12, 4)
    before = causal_running_standardize(features, history=5, minimum_history=2)
    features[8:] = -999
    after = causal_running_standardize(features, history=5, minimum_history=2)
    np.testing.assert_allclose(before[:8], after[:8])
    assert not before[:2].any()


def test_spatial_residual_normalization_preserves_statistics():
    features = np.arange(24, dtype=np.float32).reshape(2, 12)
    result = spatial_residual_standardize(
        features, lags=2, map_values=4, statistics=2
    )
    np.testing.assert_array_equal(result[:, 4:6], features[:, 4:6])
    np.testing.assert_array_equal(result[:, 10:12], features[:, 10:12])


def test_roi_transform_records_exact_coordinates_and_scale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = BehaviorRoiExtractor(output_size=(64, 32)).extract(
        frame,
        Box(20, 10, 180, 90),
        RoiStrategy.UPPER_BODY,
        source_camera="BODY",
        driver_track_id="driver-1",
    )[0]
    assert result.image.shape == (32, 64, 3)
    assert result.provenance.original_frame_box == (20, 10, 180, 90)
    assert result.provenance.crop_box == (28, 10, 172, 71)
    assert result.provenance.driver_track_id == "driver-1"


def test_dynamic_roi_preserves_phone_on_frame_boundary():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[85:100, 90:100, 0] = 255
    crop = BehaviorRoiExtractor(output_size=(64, 64), padding=0.5).extract(
        frame,
        Box(0, 0, 100, 100),
        RoiStrategy.DYNAMIC_PADDED,
        source_camera="BODY",
        phone_box=Box(90, 85, 100, 100),
    )[0]
    assert crop.image[..., 0].max() == 255
    assert crop.provenance.padding_ltrb[2] > 0
    assert "BOUNDARY_PROTECTED" in crop.provenance.reason_codes


def test_missing_hand_face_evidence_remains_unknown():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    crop = BehaviorRoiExtractor().extract(
        frame,
        Box(0, 0, 64, 64),
        RoiStrategy.FACE_HAND,
        source_camera="BODY_ONLY",
    )[0]
    assert crop.provenance.evidence_state == EvidenceState.UNKNOWN.value
    assert crop.provenance.driver_track_id == "UNKNOWN"
    assert "NO_FACE_OR_HAND_BOX" in crop.provenance.reason_codes


def test_mounted_camera_is_not_associated_with_driver():
    result = DriverPhoneAssociation().associate(
        Box(10, 10, 20, 20),
        (),
        phone_context="MOUNTED_OR_STATIC",
        phone_confidence=0.9,
        context_verified=True,
    )
    assert result.state == AssociationState.MOUNTED_OR_STATIC.value
    assert result.occupant_track_id == "UNKNOWN"


def test_verified_passenger_phone_does_not_become_driver_event():
    passenger = OccupantTrack("passenger-1", "front_passenger", Box(50, 0, 100, 100), True, 0.9)
    result = DriverPhoneAssociation().associate(
        Box(60, 20, 70, 35),
        (passenger,),
        phone_context="HANDHELD_USE",
        phone_confidence=0.8,
        hand_proximity=0.9,
    )
    assert result.state == AssociationState.PASSENGER_ASSOCIATED.value
    assert result.occupant_track_id == "passenger-1"


def test_unverified_or_missing_occupant_returns_unknown():
    unverified = OccupantTrack("candidate", "driver", Box(0, 0, 100, 100), False, 0.95)
    result = DriverPhoneAssociation().associate(
        Box(20, 20, 30, 30),
        (unverified,),
        phone_context="HANDHELD_USE",
        phone_confidence=0.9,
    )
    assert result.state == AssociationState.UNKNOWN.value
    assert result.confidence == 0.0
    assert "NO_VERIFIED_OCCUPANT_TRACK" in result.reason_codes


def test_verified_driver_requires_containment_not_nearest_distance():
    driver = OccupantTrack("driver-1", "driver", Box(0, 0, 40, 100), True, 0.9)
    result = DriverPhoneAssociation().associate(
        Box(50, 20, 60, 30),
        (driver,),
        phone_context="HANDHELD_USE",
        phone_confidence=0.9,
    )
    assert result.state == AssociationState.UNKNOWN.value
    assert "NO_OCCUPANT_CONTAINS_PHONE" in result.reason_codes


def test_v11_development_loader_rejects_protected_holdout():
    with pytest.raises(ValueError, match="not eligible"):
        _load_subject("gE-28")


def test_v11_metric_calculation_is_reproducible():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 1, 2, 0])
    first = _metrics(labels, predictions)
    second = _metrics(labels.copy(), predictions.copy())
    assert first == second
    assert first["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_dataset_rights_gate_remains_closed():
    manifest = json.loads(
        (ROOT / "reports" / "v10_research" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["rights_accepted"] is False
    assert manifest["training_allowed_by_v10_audit"] is False
    assert manifest["protected_holdouts_accessed_in_v10"] is False
