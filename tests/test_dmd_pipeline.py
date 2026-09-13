"""Unit tests for DMD adapter, timeline validation, split audit, and temporal matching."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from training.dmd.adapter import DMDInterval, DMDSessionAnnotation, parse_dmd_openlabel
from training.dmd.split import audit_dmd_split
from training.dmd.temporal_eval import (
    TemporalMatchingPolicy,
    match_temporal_events,
)


def test_dmd_adapter_parses_openlabel():
    sample_json = {
        "openlabel": {
            "metadata": {"sub_id": "14"},
            "actions": {
                "act_1": {
                    "type": "driver_actions/phonecall_right",
                    "frame_intervals": [{"frame_start": 100, "frame_end": 200}],
                },
                "act_2": {
                    "type": "driver_actions/safe_drive",
                    "frame_intervals": [{"frame_start": 201, "frame_end": 300}],
                },
                "act_3": {
                    "type": "driver_actions/texting_left",
                    "frame_intervals": [{"frame_start": 350, "frame_end": 450}],
                },
            },
        }
    }
    with TemporaryDirectory() as tmpdir:
        ann_file = Path(tmpdir) / "gC_14_s2_test_rgb_ann_distraction.json"
        ann_file.write_text(json.dumps(sample_json), encoding="utf-8")

        res = parse_dmd_openlabel(ann_file, fps=30.0, total_frames=500)
        assert res.source_participant_id == "14"
        assert res.source_group == "gC"
        assert res.source_session_id == "s2"
        assert len(res.phone_intervals) == 2
        assert len(res.all_intervals) == 3

        phone_types = [x.source_action for x in res.phone_intervals]
        assert "driver_actions/phonecall_right" in phone_types
        assert "driver_actions/texting_left" in phone_types
        assert res.phone_intervals[0].start_seconds == pytest.approx(100 / 30.0, 0.01)


def test_dmd_split_audit_zero_overlap():
    cal_manifest = [
        {
            "source_participant_id": "14",
            "member_sha256": "sha_vid_14",
            "annotation_member_path": "fake/path/14.json",
            "duration_seconds": 400.0,
        },
        {
            "source_participant_id": "37",
            "member_sha256": "sha_vid_37",
            "annotation_member_path": "fake/path/37.json",
            "duration_seconds": 380.0,
        },
    ]
    eval_manifest = [
        {
            "source_participant_id": "36",
            "member_sha256": "sha_vid_36",
            "annotation_member_path": "fake/path/36.json",
            "duration_seconds": 410.0,
        }
    ]

    res = audit_dmd_split(cal_manifest, eval_manifest)
    assert res.status == "PASS"
    assert res.subject_overlap == 0
    assert res.sha_overlap == 0
    assert set(res.calibration_subjects) == {"14", "37"}
    assert res.evaluation_subjects == ["36"]


def test_dmd_split_audit_detects_leakage():
    cal_manifest = [
        {
            "source_participant_id": "14",
            "member_sha256": "sha_shared",
            "annotation_member_path": "fake/path/14.json",
        }
    ]
    eval_manifest = [
        {
            "source_participant_id": "14",
            "member_sha256": "sha_shared",
            "annotation_member_path": "fake/path/14.json",
        }
    ]
    res = audit_dmd_split(cal_manifest, eval_manifest)
    assert res.status == "FAIL"
    assert res.subject_overlap == 1
    assert res.sha_overlap == 1


def test_temporal_event_matching_policy():
    gt = [
        {"start_seconds": 10.0, "end_seconds": 20.0, "source_action": "phonecall_right"},
        {"start_seconds": 50.0, "end_seconds": 65.0, "source_action": "texting_left"},
    ]
    predictions = [
        {"start_seconds": 10.5, "end_seconds": 19.8, "confidence": 0.85},  # Match with GT 0
        {"start_seconds": 30.0, "end_seconds": 35.0, "confidence": 0.70},  # False Alarm (FP)
        {"start_seconds": 49.5, "end_seconds": 64.0, "confidence": 0.90},  # Match with GT 1
    ]

    policy = TemporalMatchingPolicy(iou_threshold=0.3, onset_tolerance_seconds=2.0)
    matched, fps, fns = match_temporal_events(gt, predictions, policy)

    assert len(matched) == 2
    assert len(fps) == 1
    assert len(fns) == 0
    assert fps[0]["start_seconds"] == 30.0
    assert matched[0]["start_latency"] == pytest.approx(0.5, 0.01)
