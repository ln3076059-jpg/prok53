"""Build the evidence-preserving V11 audit of the frozen 13 phone false negatives."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.common import sha256_file
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _full_frame_box(selected: dict[str, Any] | None, sample: dict[str, Any] | None) -> list[float] | None:
    if not selected or not sample:
        return None
    x1, y1, x2, y2 = selected["xyxy_in_driver_roi"]
    rx1, ry1, _rx2, _ry2 = sample["driver_roi_xyxy_in_full_frame"]
    return [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1]


def _failure_stages(diagnostic: dict[str, Any], review: dict[str, Any]) -> list[str]:
    stages: list[str] = []
    if diagnostic["detector"]["samples_with_phone_box"] == 0:
        stages.append("SPECIALIST_DETECTOR_MISS")
    if review["selected_proposal"] is None:
        stages.extend(["AUXILIARY_DETECTOR_NO_VALID_BOX", "OCCLUSION_OR_LOCALIZATION_FAILURE"])
    else:
        stages.append("AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE")
    if review["occupant_track_status"] != "VERIFIED":
        stages.append("DRIVER_ASSOCIATION_UNKNOWN")
    if not diagnostic["action_probability"]["T1_reaches_gate"]:
        stages.append("ACTION_EVIDENCE_BELOW_GATE")
    if diagnostic["duration_seconds"] < 1.0:
        stages.append("SUBSECOND_TEMPORAL_RISK")
    if not stages:
        stages.append("FUSION_OR_TEMPORAL_EVENT_FAILURE")
    return stages


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    sources = {
        "diagnostics": "reports/v9_research/v9_diagnostics.json",
        "reviews": "reports/v9_research/human_phone_review_summary.json",
        "recovery": "reports/ai85_next/fn_recovery.json",
        "proposals": "reports/v9_research/phone_proposal_mining.json",
    }
    diagnostics = _read(sources["diagnostics"])["phone_false_negatives"]["cases"]
    reviews = _read(sources["reviews"])["cases"]
    recovery = _read(sources["recovery"])["cases"]
    proposals = _read(sources["proposals"])["rows"]
    review_by_id = {row["case_id"]: row for row in reviews}
    recovery_by_id = {row["case_id"]: row for row in recovery}
    proposal_by_id = {row["case_id"]: row for row in proposals}

    cases: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        case_id = diagnostic["case_id"]
        review = review_by_id[case_id]
        recovered = recovery_by_id[case_id]
        proposal_row = proposal_by_id[case_id]
        selected = review["selected_proposal"]
        selected_sample = review["selected_sample"]
        all_candidates = [
            candidate
            for sample in proposal_row["samples"]
            for candidate in sample["proposals"]
        ]
        auxiliary_max = max((float(row["confidence"]) for row in all_candidates), default=None)
        stages = _failure_stages(diagnostic, review)
        truth = recovered["truth"]
        action = diagnostic["action_probability"]
        cases.append(
            {
                "event_id": case_id,
                "subject_id": diagnostic["subject_id"],
                "ground_truth_action": diagnostic["action"],
                "timestamp": {
                    "start_seconds": truth["start_seconds"],
                    "end_seconds": truth["end_seconds"],
                    "start_frame": truth["start_frame"],
                    "end_frame": truth["end_frame"],
                },
                "phone_visibility": review["phone_visibility"],
                "phone_use_assessment": review["phone_use_assessment"],
                "object_detection": {
                    "specialist_raw_max_confidence": diagnostic["detector"]["maximum_phone_confidence"],
                    "specialist_samples_with_box": diagnostic["detector"]["samples_with_phone_box"],
                    "auxiliary_max_confidence": auxiliary_max,
                    "human_selected_auxiliary_confidence": selected["confidence"] if selected else None,
                    "selected_proposal_id": review["selected_proposal_id"],
                    "selected_box_in_driver_roi": selected["xyxy_in_driver_roi"] if selected else None,
                    "selected_box_in_original_frame": _full_frame_box(selected, selected_sample),
                    "driver_roi_in_original_frame": (
                        selected_sample["driver_roi_xyxy_in_full_frame"] if selected_sample else None
                    ),
                    "candidate_count": len(all_candidates),
                    "candidate_is_event_recovery": False,
                },
                "hand_face_context": {
                    "geometry_context": selected["geometry_context"] if selected else None,
                    "hand_proximity": selected["hand_proximity"] if selected else None,
                    "face_proximity": selected["face_proximity"] if selected else None,
                    "visual_review": diagnostic["visual_review"],
                    "human_notes": review["notes"],
                },
                "driver_association": {
                    "state": "UNKNOWN",
                    "source_status": review["occupant_track_status"],
                    "reason_codes": ["NO_VERIFIED_OCCUPANT_TRACK"],
                },
                "action_evidence": {
                    "fixed_gate": action["fixed_gate"],
                    "T0_max": action["T0_max"],
                    "T1_dense_max": action["T1_dense_max"],
                    "T0_reaches_gate": action["T0_reaches_gate"],
                    "T1_reaches_gate": action["T1_reaches_gate"],
                    "frozen_baseline_timeline": recovered["baseline_trace"].get("trace", []),
                    "timeline_summary": {
                        key: value
                        for key, value in recovered["baseline_trace"].items()
                        if key != "trace"
                    },
                },
                "gate_rejection_reason": diagnostic["root_cause"],
                "event_creation_state": "NO_MATCHED_FINAL_EVENT",
                "potential_missed_interval": {
                    "start_seconds": truth["start_seconds"],
                    "end_seconds": truth["end_seconds"],
                    "interpretation": "GROUND_TRUTH_INTERVAL_ONLY_NOT_A_PREDICTED_BOUNDARY",
                },
                "failure_stages": stages,
                "match_result": "FALSE_NEGATIVE",
                "review_receipt": {
                    "path": review["receipt_path"],
                    "sha256": review["receipt_sha256"],
                },
            }
        )

    if len(cases) != 13:
        raise ValueError(f"Expected frozen 13-case population, got {len(cases)}")
    if {case["subject_id"] for case in cases} - set(DEVELOPMENT_SUBJECTS):
        raise ValueError("Unexpected subject in frozen false-negative set")
    if set(DEVELOPMENT_SUBJECTS) & set(PROTECTED_HOLDOUTS):
        raise ValueError("Development and protected subject lists overlap")

    stage_counts = Counter(stage for case in cases for stage in case["failure_stages"])
    result = {
        "schema_version": 2,
        "evidence_level": "LEVEL_1_DIAGNOSTIC_WITH_AUTHENTICATED_HUMAN_REVIEW",
        "scope": "FROZEN_13_REPEATED_DEVELOPMENT_FALSE_NEGATIVES",
        "development_subjects": sorted(DEVELOPMENT_SUBJECTS),
        "protected_holdouts": sorted(PROTECTED_HOLDOUTS),
        "protected_holdouts_accessed": False,
        "source_artifacts": {path: sha256_file(ROOT / path) for path in sources.values()},
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "phone_visible": sum(case["phone_visibility"] == "PHONE_VISIBLE" for case in cases),
            "valid_auxiliary_box": sum(
                case["object_detection"]["selected_box_in_driver_roi"] is not None for case in cases
            ),
            "verified_occupant_tracks": 0,
            "T0_reaches_gate": sum(case["action_evidence"]["T0_reaches_gate"] for case in cases),
            "T1_reaches_gate": sum(case["action_evidence"]["T1_reaches_gate"] for case in cases),
            "failure_stage_counts_nonexclusive": dict(sorted(stage_counts.items())),
        },
        "ground_truth_changed": False,
        "event_matching_policy_changed": False,
        "recovered_events_claimed": 0,
        "fine_tuning_performed": False,
        "production_approved": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fn_root_cause.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
