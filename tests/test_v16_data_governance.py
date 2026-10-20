from __future__ import annotations

import pytest
import yaml

from training.data_governance import (
    UnifiedAnnotation,
    require_download_authorized,
    require_training_authorized,
    validate_bbox,
    validate_group_split,
)
from training.download_source import load_source


def test_download_requires_explicit_authorized_status_and_operation() -> None:
    candidate = {
        "source_id": "candidate",
        "status": "DISCOVERED",
        "license": "CC-BY-4.0",
        "allowed_operations": ["download"],
    }
    with pytest.raises(PermissionError, match="DOWNLOAD_AUTHORIZED"):
        require_download_authorized(candidate)
    require_download_authorized({**candidate, "status": "DOWNLOAD_AUTHORIZED"})


def test_download_rejects_unverified_license() -> None:
    with pytest.raises(PermissionError, match="license"):
        require_download_authorized(
            {
                "source_id": "candidate",
                "status": "DOWNLOAD_AUTHORIZED",
                "license": "UNKNOWN",
                "allowed_operations": ["download"],
            }
        )


def test_downloader_fails_closed_before_network_access(tmp_path) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "source_id": "public-link-is-not-authorization",
                        "status": "CANDIDATE_REVIEW_REQUIRED",
                        "license": "CC-BY-4.0",
                        "allowed_operations": ["download"],
                        "download": {"type": "direct_zip", "url": "https://example.invalid/x"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="DOWNLOAD_AUTHORIZED"):
        load_source(registry, "public-link-is-not-authorization")


def test_training_requires_reviewed_status_and_explicit_operation() -> None:
    source = {
        "source_id": "review-pending",
        "status": "ANNOTATION_PENDING",
        "license": "CC-BY-4.0",
        "allowed_operations": ["training"],
    }
    with pytest.raises(PermissionError, match="APPROVED_FOR_TRAINING"):
        require_training_authorized(source)
    require_training_authorized({**source, "status": "APPROVED_FOR_TRAINING"})


def test_group_split_detects_subject_and_augmented_origin_leakage() -> None:
    records = [
        {"split": "train", "trusted_subject_id": "s1", "original_asset_id": "origin"},
        {"split": "test", "trusted_subject_id": "s1", "original_asset_id": "origin"},
    ]
    result = validate_group_split(records)
    assert result["status"] == "FAIL"
    assert result["overlaps"]["trusted_subject_id"] == ["s1"]
    assert result["overlaps"]["original_asset_id"] == ["origin"]


def test_group_split_blocks_protected_subject_from_fit() -> None:
    result = validate_group_split([{"split": "train", "trusted_subject_id": "gZ-37"}])
    assert result["status"] == "FAIL"
    assert result["protected_subjects_in_fit"] == ["gZ-37"]


def test_bbox_validation_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="positive area"):
        validate_bbox((0.5, 0.2, 0.4, 0.8))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_bbox((0.1, 0.2, 1.2, 0.8))


def test_static_annotation_cannot_invent_event_timestamps() -> None:
    annotation = UnifiedAnnotation(
        source_id="statefarm",
        source_asset_id="img.jpg",
        source_hash="a" * 64,
        original_label="c1",
        mapping_version="v1",
        review_status="PENDING",
        start_timestamp=1.0,
        end_timestamp=2.0,
    )
    with pytest.raises(ValueError, match="timestamps require a video"):
        annotation.validate()


def test_visible_phone_is_not_automatically_safe() -> None:
    annotation = UnifiedAnnotation(
        source_id="fixture",
        source_asset_id="img.jpg",
        source_hash="b" * 64,
        original_label="unknown",
        mapping_version="v1",
        review_status="PENDING",
        phone_visibility="PHONE_VISIBLE",
        phone_action="SAFE_DRIVING",
        phone_context="HANDHELD",
    )
    with pytest.raises(ValueError, match="SAFE_DRIVING"):
        annotation.validate()


def test_approved_annotation_requires_reviewer() -> None:
    annotation = UnifiedAnnotation(
        source_id="fixture",
        source_asset_id="img.jpg",
        source_hash="c" * 64,
        original_label="unclear",
        mapping_version="v1",
        review_status="APPROVED",
    )
    with pytest.raises(ValueError, match="reviewer"):
        annotation.validate()
