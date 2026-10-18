from __future__ import annotations

import pytest

from training.dmd.candidate_flow import (
    CandidateDecision,
    CandidateFlowLedger,
)
from training.dmd.phone_benchmark import (
    PhoneBenchmarkImage,
    PhoneBox,
    validate_phone_benchmark,
)
from training.split_dataset import assign_groups, validate_isolation


def approved(group: str, **values: object) -> dict:
    return {
        "sample_id": group,
        "review_status": "APPROVED",
        "source_group_id": group,
        **values,
    }


def test_split_keeps_all_groups_from_one_subject_together() -> None:
    records = [
        approved("g1", trusted_subject_id="subject-1"),
        approved("g2", trusted_subject_id="subject-1"),
        approved("g3", trusted_subject_id="subject-2"),
        approved("g4", trusted_subject_id="subject-3"),
    ]
    assignments = assign_groups(records)
    assert assignments["g1"] == assignments["g2"]


def test_split_keeps_original_and_augmentation_together() -> None:
    records = [
        approved("original", original_asset_id="asset-1"),
        approved("augmented", original_asset_id="asset-1"),
        approved("other", original_asset_id="asset-2"),
    ]
    assignments = assign_groups(records)
    assert assignments["original"] == assignments["augmented"]


def test_split_rejects_protected_subject_before_materialization() -> None:
    with pytest.raises(ValueError, match="protected subject"):
        assign_groups([approved("protected", trusted_subject_id="gZ-37")])


def test_isolation_detects_video_and_original_asset_leakage() -> None:
    report = validate_isolation(
        [
            {
                "split": "train",
                "video_id": "video-1",
                "original_asset_id": "asset-1",
                "source_group_id": "a",
            },
            {
                "split": "test",
                "video_id": "video-1",
                "original_asset_id": "asset-1",
                "source_group_id": "b",
            },
        ]
    )
    assert report["status"] == "FAIL"
    assert report["overlaps"]["video_id"] == ["video-1"]
    assert report["overlaps"]["original_asset_id"] == ["asset-1"]


def positive_record(digest: str = "a" * 64) -> PhoneBenchmarkImage:
    return PhoneBenchmarkImage(
        image_sha256=digest,
        source_id="authorized-source",
        source_asset_id="positive.jpg",
        source_license="VERIFIED-RESEARCH-LICENSE",
        width=640,
        height=480,
        context="HANDHELD",
        reviewer_id="reviewer-1",
        review_status="APPROVED",
        annotation_scope="ALL_VISIBLE_PHONES",
        negative_scope_reviewed=False,
        boxes=(PhoneBox((100, 100, 130, 160), "VISIBLE", "DRIVER", 0.9),),
    )


def negative_record(digest: str = "b" * 64) -> PhoneBenchmarkImage:
    return PhoneBenchmarkImage(
        image_sha256=digest,
        source_id="authorized-source",
        source_asset_id="negative.jpg",
        source_license="VERIFIED-RESEARCH-LICENSE",
        width=640,
        height=480,
        context="MOUNTED_CAMERA",
        reviewer_id="reviewer-2",
        review_status="APPROVED",
        annotation_scope="ALL_VISIBLE_PHONES",
        negative_scope_reviewed=True,
    )


def test_phone_benchmark_requires_reviewed_zero_box_negatives() -> None:
    invalid = PhoneBenchmarkImage(
        **{**negative_record().__dict__, "negative_scope_reviewed": False}
    )
    with pytest.raises(ValueError, match="negative-scope review"):
        invalid.validate()


def test_phone_benchmark_requires_positive_and_negative_truth() -> None:
    blocked = validate_phone_benchmark([positive_record()])
    assert blocked["status"] == "BLOCKED"
    assert blocked["blockers"] == ["NO_REVIEWED_NEGATIVE_IMAGES"]
    ready = validate_phone_benchmark([positive_record(), negative_record()])
    assert ready["status"] == "READY"
    assert ready["phone_boxes"] == 1


def test_phone_benchmark_rejects_duplicate_content() -> None:
    with pytest.raises(ValueError, match="duplicate image hash"):
        validate_phone_benchmark([positive_record(), negative_record("a" * 64)])


def test_candidate_flow_reports_gt_rejection_and_negative_admission() -> None:
    ledger = CandidateFlowLedger()
    ledger.record(CandidateDecision("positive", "DETECTED", True, "PHONE_BOX", "POSITIVE"))
    ledger.record(
        CandidateDecision(
            "positive", "DRIVER_ASSOCIATED", False, "NO_VERIFIED_DRIVER", "POSITIVE"
        )
    )
    ledger.record(CandidateDecision("negative", "DETECTED", True, "PHONE_BOX", "NEGATIVE"))
    summary = ledger.summary()
    assert summary["ground_truth_positive_rejections"] == 1
    assert summary["negative_candidates_admitted"] == 1
    assert summary["rejection_reasons"] == {"NO_VERIFIED_DRIVER": 1}


def test_candidate_flow_rejects_duplicate_or_backward_stage() -> None:
    ledger = CandidateFlowLedger()
    ledger.record(CandidateDecision("c1", "DRIVER_ASSOCIATED", True, "VERIFIED", "UNKNOWN"))
    with pytest.raises(ValueError, match="forward order"):
        ledger.record(CandidateDecision("c1", "DETECTED", True, "BOX", "UNKNOWN"))
