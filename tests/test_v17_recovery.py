from __future__ import annotations

import pytest

from training.dmd.candidate_flow import (
    CandidateDecision,
    CandidateFlowLedger,
)
from training.dmd.phone_benchmark import (
    DriverAssociationBenchmarkRecord,
    PhoneBenchmarkImage,
    PhoneBox,
    PhoneEventBenchmarkRecord,
    SeatbeltBenchmarkRecord,
    evaluate_phone_benchmark,
    validate_phone_benchmark,
    validate_phone_event_benchmark,
    validate_seatbelt_benchmark,
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


def test_phone_box_area_and_small_phone_detection() -> None:
    small_box = PhoneBox((10, 10, 30, 30), "VISIBLE", "DRIVER")
    assert small_box.area == 400.0
    assert small_box.is_small is True

    large_box = PhoneBox((10, 10, 100, 100), "VISIBLE", "DRIVER")
    assert large_box.area == 8100.0
    assert large_box.is_small is False


def test_seatbelt_benchmark_requires_three_states_and_human_review() -> None:
    record_fastened = SeatbeltBenchmarkRecord(
        sample_id="sb-01",
        image_sha256="c" * 64,
        source_id="reviewed-src",
        source_license="CC-BY-4.0",
        torso_xyxy=(100, 100, 300, 500),
        state="FASTENED",
        reviewer_id="human-reviewer",
        review_status="APPROVED",
    )
    record_unfastened = SeatbeltBenchmarkRecord(
        sample_id="sb-02",
        image_sha256="d" * 64,
        source_id="reviewed-src",
        source_license="CC-BY-4.0",
        torso_xyxy=(100, 100, 300, 500),
        state="UNFASTENED",
        reviewer_id="human-reviewer",
        review_status="APPROVED",
    )
    record_uncertain = SeatbeltBenchmarkRecord(
        sample_id="sb-03",
        image_sha256="e" * 64,
        source_id="reviewed-src",
        source_license="CC-BY-4.0",
        torso_xyxy=(100, 100, 300, 500),
        state="UNCERTAIN_OR_OCCLUDED",
        reviewer_id="human-reviewer",
        review_status="APPROVED",
        occlusion_reason="DARK_JACKET_AND_HAIR_OCCLUSION",
    )
    result = validate_seatbelt_benchmark([record_fastened, record_unfastened, record_uncertain])
    assert result["status"] == "READY"
    assert result["total_records"] == 3

    # Occluded without reason must be rejected
    with pytest.raises(ValueError, match="explicit occlusion_reason"):
        SeatbeltBenchmarkRecord(
            sample_id="sb-04",
            image_sha256="f" * 64,
            source_id="reviewed-src",
            source_license="CC-BY-4.0",
            torso_xyxy=(100, 100, 300, 500),
            state="UNCERTAIN_OR_OCCLUDED",
            reviewer_id="human-reviewer",
            review_status="APPROVED",
            occlusion_reason=None,
        ).validate()


def test_phone_event_benchmark_rejects_synthetic_static_image_timestamps() -> None:
    with pytest.raises(ValueError, match="synthetic timestamps from static images"):
        PhoneEventBenchmarkRecord(
            event_id="ev-01",
            video_id="vid-01",
            source_id="src-01",
            source_license="CC-BY-4.0",
            start_timestamp=10.0,
            end_timestamp=15.0,
            event_type="PHONE_USE",
            camera_id="cam-01",
            reviewer_id="human-reviewer",
            review_status="APPROVED",
            is_static_image_synthetic=True,
        ).validate()


def test_phone_event_benchmark_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="strictly greater than start_timestamp"):
        PhoneEventBenchmarkRecord(
            event_id="ev-02",
            video_id="vid-01",
            source_id="src-01",
            source_license="CC-BY-4.0",
            start_timestamp=15.0,
            end_timestamp=10.0,
            event_type="PHONE_USE",
            camera_id="cam-01",
            reviewer_id="human-reviewer",
            review_status="APPROVED",
        ).validate()

    ev1 = PhoneEventBenchmarkRecord(
        event_id="ev-03",
        video_id="vid-01",
        source_id="src-01",
        source_license="CC-BY-4.0",
        start_timestamp=10.0,
        end_timestamp=14.5,
        event_type="PHONE_USE",
        camera_id="cam-01",
        reviewer_id="human-reviewer",
        review_status="APPROVED",
    )
    ev2 = PhoneEventBenchmarkRecord(
        event_id="ev-04",
        video_id="vid-01",
        source_id="src-01",
        source_license="CC-BY-4.0",
        start_timestamp=20.0,
        end_timestamp=30.0,
        event_type="SAFE_DRIVING",
        camera_id="cam-01",
        reviewer_id="human-reviewer",
        review_status="APPROVED",
    )
    res = validate_phone_event_benchmark([ev1, ev2])
    assert res["status"] == "READY"
    assert res["total_events"] == 2
    assert res["total_duration_seconds"] == 14.5


def test_driver_association_benchmark_requires_approved_occupant_role() -> None:
    rec = DriverAssociationBenchmarkRecord(
        sample_id="assoc-01",
        image_sha256="a" * 64,
        occupant_track_id="trk-1",
        true_role="DRIVER",
        phone_box_xyxy=(100, 100, 150, 180),
        occupant_box_xyxy=(80, 50, 400, 450),
        association_confidence=0.95,
        reviewer_id="human-reviewer",
        review_status="APPROVED",
    )
    rec.validate()

    with pytest.raises(ValueError, match="invalid occupant role"):
        DriverAssociationBenchmarkRecord(
            sample_id="assoc-02",
            image_sha256="a" * 64,
            occupant_track_id="trk-1",
            true_role="INVALID_ROLE",
            phone_box_xyxy=(100, 100, 150, 180),
            occupant_box_xyxy=(80, 50, 400, 450),
            association_confidence=0.95,
            reviewer_id="human-reviewer",
            review_status="APPROVED",
        ).validate()


def test_evaluate_phone_benchmark_fails_closed_without_approved_data() -> None:
    eval_res = evaluate_phone_benchmark([positive_record()])
    assert eval_res["status"] == "BLOCKED"
    assert eval_res["ap_metrics_allowed"] is False
    assert "NO_REVIEWED_NEGATIVE_IMAGES" in eval_res["blockers"]

