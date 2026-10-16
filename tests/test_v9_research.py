from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.dmd.v9_diagnostics import (
    assert_development_subject,
    classify_fn_bottleneck,
    summarize_detection_row,
)


def test_missing_phone_box_is_explicit_and_blocks_geometry():
    summary = summarize_detection_row(
        {
            "samples": [
                {"decode_status": "OK", "phone_proposals": []},
                {"decode_status": "OK", "phone_proposals": []},
            ]
        }
    )
    assert summary["phone_box_status"] == "MISSING_ALL_SAMPLED_FRAMES"
    assert summary["occupant_id_status"] == "BLOCKED_NO_PHONE_BOX"
    assert summary["geometry_status"] == "BLOCKED_NO_PHONE_BOX"
    assert classify_fn_bottleneck(summary, 0.95) == "PHONE_DETECTOR_NO_PROPOSAL"


def test_roi_level_phone_box_does_not_claim_occupant_identity():
    summary = summarize_detection_row(
        {
            "samples": [
                {
                    "decode_status": "OK",
                    "phone_proposals": [
                        {
                            "confidence": 0.42,
                            "occupant_association": "DRIVER_ROI_LEVEL_ONLY_NO_TRACK_ID",
                        }
                    ],
                },
                {"decode_status": "OK", "phone_proposals": []},
            ]
        }
    )
    assert summary["phone_box_status"] == "AVAILABLE_ON_SUBSET"
    assert summary["occupant_id_status"] == "MISSING_TRACK_ID"
    assert summary["geometry_status"] == "UNVERIFIED_ROI_LEVEL_ASSOCIATION"
    assert classify_fn_bottleneck(summary, 0.55) == "PHONE_BOX_PRESENT_ACTION_UNDER_GATE"


@pytest.mark.parametrize("subject", ["gZ-37", "gE-28", "unknown"])
def test_v9_rejects_protected_or_unregistered_subjects(subject: str):
    with pytest.raises(ValueError, match="not registered"):
        assert_development_subject(subject)


def test_v9_source_artifacts_share_exact_thirteen_fn_cases():
    evidence = json.loads(Path("reports/ai85_plus/evidence_diagnostics.json").read_text())
    detector = json.loads(Path("reports/ai85_next/detector_trace.json").read_text())
    manifest = json.loads(Path("reports/ai85_upgrade/fn_frame_manifest.json").read_text())
    expected = {item["case_id"] for item in evidence["false_negative_cases"]}
    assert len(expected) == 13
    assert expected == {item["case_id"] for item in detector["rows"]}
    assert expected == {item["case_id"] for item in manifest}
    assert not any(name in json.dumps(evidence) for name in ("gZ-37", "gE-28"))
