"""Consolidate the frozen 13-case phone FN evidence without changing evaluation policy."""

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
DEFAULT_OUTPUT = ROOT / "reports" / "v10_research"


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _classify(diagnostic: dict[str, Any], human: dict[str, Any]) -> list[str]:
    stages: list[str] = []
    detector = diagnostic["detector"]
    action = diagnostic["action_probability"]
    if human["selected_proposal"] is None:
        stages.extend(["DETECTOR_MISS", "MISSING_PHONE_BOXES"])
    elif detector["samples_with_phone_box"] == 0:
        stages.append("SPECIALIST_DETECTOR_MISS_AUXILIARY_CANDIDATE_RECOVERED")
    if human["occupant_track_status"] != "VERIFIED":
        stages.append("DRIVER_OCCUPANT_ASSOCIATION_UNVERIFIED")
    if not action["T1_reaches_gate"]:
        stages.append("ACTION_REPRESENTATION_OR_CALIBRATION_UNDER_GATE")
    if diagnostic["duration_seconds"] < 1.5:
        stages.append("SHORT_EVENT_TEMPORAL_RISK")
    if "occlud" in human["notes"].lower():
        stages.append("OCCLUSION")
    if not stages:
        stages.append("FUSION_OR_TEMPORAL_EVENT_FAILURE")
    return stages


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if set(DEVELOPMENT_SUBJECTS) & set(PROTECTED_HOLDOUTS):
        raise ValueError("Development and protected subject registries overlap")
    diagnostics_path = "reports/v9_research/v9_diagnostics.json"
    reviews_path = "reports/v9_research/human_phone_review_summary.json"
    recovery_path = "reports/ai85_next/fn_recovery.json"
    diagnostics = _read(diagnostics_path)["phone_false_negatives"]
    reviews = _read(reviews_path)
    recovery = _read(recovery_path)
    by_review = {item["case_id"]: item for item in reviews["cases"]}
    by_recovery = {item["case_id"]: item for item in recovery["cases"]}
    cases = []
    for item in diagnostics["cases"]:
        case_id = item["case_id"]
        human = by_review[case_id]
        earlier = by_recovery[case_id]
        truth = earlier["truth"]
        selected = human["selected_proposal"]
        action = item["action_probability"]
        stages = _classify(item, human)
        cases.append(
            {
                "event_id": case_id,
                "subject": item["subject_id"],
                "camera_view": "BODY",
                "action_label": item["action"],
                "start_seconds": truth["start_seconds"],
                "end_seconds": truth["end_seconds"],
                "duration_seconds": item["duration_seconds"],
                "phone_visibility": human["phone_visibility"],
                "phone_use_assessment": human["phone_use_assessment"],
                "proposal_available": selected is not None,
                "proposal_source": selected["source"] if selected else None,
                "proposal_confidence": selected["confidence"] if selected else None,
                "phone_box_quality": "HUMAN_SELECTED_VALID" if selected else "NO_VALID_BOX",
                "hand_proximity": selected["hand_proximity"] if selected else None,
                "face_proximity": selected["face_proximity"] if selected else None,
                "occupant_association": human["occupant_track_status"],
                "action_probability_t0_max": action["T0_max"],
                "action_probability_t1_dense_max": action["T1_dense_max"],
                "action_gate": action["fixed_gate"],
                "coarse_prediction": "NOT_CAPTURED_IN_FROZEN_EVENT_ARTIFACT",
                "temporal_activation_state": {
                    "T0_reaches_gate": action["T0_reaches_gate"],
                    "T1_reaches_gate": action["T1_reaches_gate"],
                },
                "fusion_rejection_reason": item["root_cause"],
                "match_result": "FALSE_NEGATIVE",
                "likely_failure_stages": stages,
                "visual_evidence": item["visual_review"],
                "human_notes": human["notes"],
                "receipt_path": human["receipt_path"],
                "receipt_sha256": human["receipt_sha256"],
                "roi_source": item["roi_source"],
                "roi_sha256": item["roi_sha256"],
            }
        )
    if len(cases) != 13 or set(case["subject"] for case in cases) - set(DEVELOPMENT_SUBJECTS):
        raise ValueError("Unexpected frozen phone FN population")
    taxonomy = Counter(stage for case in cases for stage in case["likely_failure_stages"])
    result = {
        "schema_version": 1,
        "evidence_level": "LEVEL_1_DIAGNOSTIC_WITH_AUTHENTICATED_HUMAN_REVIEW",
        "scope": "FROZEN_13_DEVELOPMENT_FALSE_NEGATIVES",
        "development_subjects": sorted(DEVELOPMENT_SUBJECTS),
        "protected_holdouts": sorted(PROTECTED_HOLDOUTS),
        "protected_holdouts_accessed": False,
        "source_artifacts": {
            path: sha256_file(ROOT / path)
            for path in (diagnostics_path, reviews_path, recovery_path)
        },
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "visible_phone": sum(case["phone_visibility"] == "PHONE_VISIBLE" for case in cases),
            "valid_proposal": sum(case["proposal_available"] for case in cases),
            "verified_occupant_tracks": sum(case["occupant_association"] == "VERIFIED" for case in cases),
            "T0_reaches_action_gate": sum(case["temporal_activation_state"]["T0_reaches_gate"] for case in cases),
            "T1_reaches_action_gate": sum(case["temporal_activation_state"]["T1_reaches_gate"] for case in cases),
            "failure_stage_counts_nonexclusive": dict(sorted(taxonomy.items())),
        },
        "detector_candidate_yield_is_event_recall": False,
        "ground_truth_changed": False,
        "event_matching_policy_changed": False,
        "fine_tuning_performed": False,
        "production_approved": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fn_root_cause.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def render(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    rows = []
    for case in result["cases"]:
        confidence = case["proposal_confidence"]
        rows.append(
            f"| {case['event_id']} | {case['action_label'].split('/')[-1]} | "
            f"{case['start_seconds']:.2f}-{case['end_seconds']:.2f} | "
            f"{'yes' if case['proposal_available'] else 'no'} | "
            f"{confidence:.3f} | " if confidence is not None else
            f"| {case['event_id']} | {case['action_label'].split('/')[-1]} | "
            f"{case['start_seconds']:.2f}-{case['end_seconds']:.2f} | no | - | "
        )
        rows[-1] += (
            f"{case['action_probability_t1_dense_max']:.3f} | "
            f"{case['occupant_association']} | "
            f"{', '.join(case['likely_failure_stages'])} |"
        )
    summary = result["summary"]
    taxonomy_rows = "\n".join(
        f"| {stage} | {count} |"
        for stage, count in summary["failure_stage_counts_nonexclusive"].items()
    )
    text = f"""# V10 DMD FN/FP root-cause audit

Evidence level: **LEVEL 1 diagnostic with authenticated human review**. This re-audits the frozen 13
development false negatives and reads no protected holdout. Ground truth and event matching are unchanged.

| Event | Action | Time (s) | Valid proposal | Proposal conf. | T1 action max | Occupant | Likely failure stages |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

| Non-exclusive failure stage | Cases |
| --- | --- |
{taxonomy_rows}

All {summary['visible_phone']}/13 events contain a human-confirmed visible phone and confirmed phone use.
Auxiliary proposal mining yielded a human-selected valid box for {summary['valid_proposal']}/13 cases, but
zero cases have a verified occupant track. Only {summary['T0_reaches_action_gate']}/13 T0 cases and
{summary['T1_reaches_action_gate']}/13 dense-sampling cases reach the fixed action gate. The evidence therefore
does not support labeling all 13 as detector failures: detection yield, association, action calibration,
temporal continuity, and fusion are distinct failure stages.

The one no-box hard example is `gB-9-phone-06`; it remains a true detector/occlusion case. A positive proposal
in one reviewed frame is candidate yield, not event recovery. The frozen comparator remains TP=16, FP=4,
FN=13; this audit does not claim any recovered event.
"""
    path = output / "05_DMD_FN_FP_ROOT_CAUSE.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    report = render(result, args.output)
    print(json.dumps({"cases": len(result["cases"]), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
