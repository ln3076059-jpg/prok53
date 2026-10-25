"""Prepare and validate a human-review manifest for V31 FN/FP component truth."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common import sha256_file

PHONE_STATES = ("HANDHELD_USE", "MOUNTED_OR_STATIC", "ABSENT", "UNRESOLVED")
ASSOCIATIONS = (
    "DRIVER_ASSOCIATED",
    "PASSENGER_ASSOCIATED",
    "UNKNOWN",
    "UNRESOLVED",
)
PROPOSAL_QUALITY = (
    "CORRECT_PHONE",
    "WRONG_OBJECT",
    "MIXED",
    "NO_PROPOSAL",
    "UNRESOLVED",
)
FAILURE_LAYERS = (
    "PROPOSAL_MISS",
    "PROPOSAL_LOCALIZATION",
    "ASSOCIATION",
    "CONTEXT",
    "SAMPLING_TEMPORAL_SUPPORT",
    "TEMPORAL_LIFECYCLE",
    "ANNOTATION_OR_AMBIGUOUS",
    "NONE_OR_UNRESOLVED",
)


def build_manifest(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for row in source["cases"]:
        sheet = Path(row["contact_sheet"])
        cases.append(
            {
                "case_key": f"{row['subject_id']}::{row['case_id']}",
                "subject_id": row["subject_id"],
                "case_id": row["case_id"],
                "kind": row["kind"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "candidate_count": row["candidate_count"],
                "sampled_timestamps": row["sampled_timestamps"],
                "contact_sheet": row["contact_sheet"],
                "contact_sheet_sha256": sha256_file(sheet),
                "review": {
                    "status": "PENDING",
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "phone_state": "UNRESOLVED",
                    "driver_association": "UNRESOLVED",
                    "proposal_quality": "UNRESOLVED",
                    "primary_failure_layer": "NONE_OR_UNRESOLVED",
                    "confidence_1_to_3": None,
                    "notes": "",
                },
            }
        )
    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PENDING_HUMAN_REVIEW",
        "scope": "DEVELOPMENT_ERROR_CASES_ONLY_NOT_MODEL_SELECTION_GROUND_TRUTH",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "case_count": len(cases),
        "required_reviewers_per_case": 1,
        "vocabulary": {
            "phone_state": list(PHONE_STATES),
            "driver_association": list(ASSOCIATIONS),
            "proposal_quality": list(PROPOSAL_QUALITY),
            "primary_failure_layer": list(FAILURE_LAYERS),
            "confidence_1_to_3": [1, 2, 3],
        },
        "usage_constraints": [
            "Do not mark this manifest complete without a named human reviewer and timestamp for every case.",
            "Do not use UNRESOLVED cases as supervised targets.",
            "These nine selected error cases are diagnostic and cannot estimate population-level component accuracy.",
            "Do not use this development error set as an independent holdout.",
        ],
        "cases": cases,
    }


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    reviewed = 0
    usable = 0
    seen: set[str] = set()
    for row in payload.get("cases", []):
        key = str(row.get("case_key", ""))
        if not key or key in seen:
            errors.append(f"duplicate or missing case_key: {key!r}")
        seen.add(key)
        if not Path(row.get("contact_sheet", "")).is_file():
            errors.append(f"missing contact sheet: {key}")
        review = row.get("review", {})
        if review.get("phone_state") not in PHONE_STATES:
            errors.append(f"invalid phone_state: {key}")
        if review.get("driver_association") not in ASSOCIATIONS:
            errors.append(f"invalid driver_association: {key}")
        if review.get("proposal_quality") not in PROPOSAL_QUALITY:
            errors.append(f"invalid proposal_quality: {key}")
        if review.get("primary_failure_layer") not in FAILURE_LAYERS:
            errors.append(f"invalid primary_failure_layer: {key}")
        if review.get("status") == "REVIEWED":
            reviewed += 1
            required = (
                review.get("reviewer_id"),
                review.get("reviewed_at"),
                review.get("confidence_1_to_3") in (1, 2, 3),
            )
            resolved = (
                review.get("phone_state") != "UNRESOLVED"
                and review.get("driver_association") != "UNRESOLVED"
                and review.get("proposal_quality") != "UNRESOLVED"
                and review.get("primary_failure_layer") != "NONE_OR_UNRESOLVED"
            )
            if all(required) and resolved:
                usable += 1
            else:
                errors.append(f"incomplete REVIEWED case: {key}")
        elif review.get("status") != "PENDING":
            errors.append(f"invalid review status: {key}")
    total = len(payload.get("cases", []))
    complete = total > 0 and usable == total and not errors
    return {
        "status": "PASS_COMPLETE" if complete else "PENDING_OR_INVALID",
        "case_count": total,
        "reviewed_cases": reviewed,
        "usable_reviewed_cases": usable,
        "errors": errors,
        "training_eligible": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reports/v29_ai85_recovery/visual_audit_baseline/audit_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v31_ai85_plan_execution/component_truth_manifest.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        if args.output.exists():
            raise FileExistsError("component truth manifest exists; do not overwrite review work")
        payload = build_manifest(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validate_manifest(payload), indent=2))


if __name__ == "__main__":
    main()
