from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file, stable_json_hash
from training.pretrain_review_v2 import MIN_UNCERTAIN_COUNTS
from training.uncertain_visual_decisions import (
    VISUAL_EVIDENCE_LEVEL,
    VISUAL_REVIEWER_MODEL,
)

SPLITS = ("train", "val", "test")

# Indices refer to the immutable per-page JSON files emitted together with the
# visually inspected contact sheets. These are resolved to candidate IDs below;
# downstream code never depends on an index or sort order.
SELECTED_INDICES = {
    "train": [
        2, 3, 4, 10, 17, 18, 23, 31, 34, 36, 41, 43, 49, 56, 58, 70, 73, 74,
        86, 87, 88, 90, 99, 100, 101, 104, 105, 106, 107, 108, 111, 114, 115,
        117, 119, 120, 122, 123, 125, 127, 130, 131, 135, 141, 143, 153, 159,
        165, 166, 168, 169, 172, 173, 174, 178, 179, 184, 185, 192, 193, 199,
        201, 202, 203, 208, 210, 214, 216, 219, 220, 222, 223, 227, 229, 230,
        231, 237, 239, 242, 243, 245, 248, 252, 262, 266, 269, 270, 271, 272,
        278, 279, 280, 281, 284, 286, 288, 289, 291, 295, 297,
    ],
    "val": [
        2, 6, 9, 15, 17, 18, 22, 36, 37, 40, 42, 45, 57, 58, 62, 64, 65,
        67, 68, 72, 80, 81, 85, 91, 95,
    ],
    "test": [
        1, 2, 4, 5, 7, 8, 9, 10, 14, 15, 17, 18, 19, 20, 21, 23, 29, 30,
        32, 37, 39, 40, 42, 44, 46,
    ],
}


def _contact_rows(root: Path, split: str) -> dict[int, dict]:
    rows = []
    for path in sorted(root.glob(f"{split}_page_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    by_index = {int(item["index"]): item for item in rows}
    if len(by_index) != len(rows):
        raise ValueError(f"duplicate contact-sheet index in {split}")
    return by_index


def finalize(source_review: Path, contact_root: Path, output: Path) -> dict:
    source = json.loads(source_review.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise ValueError(f"source uncertainty review did not pass: {source_review}")
    source_by_candidate = {str(item["candidate_id"]): item for item in source["samples"]}
    if len(source_by_candidate) != len(source["samples"]):
        raise ValueError("duplicate candidate_id in source uncertainty review")

    decisions = []
    existing_eligible = 0
    for split in SPLITS:
        rows = _contact_rows(contact_root, split)
        indices = SELECTED_INDICES[split]
        if len(indices) != MIN_UNCERTAIN_COUNTS[split] or len(set(indices)) != len(indices):
            raise ValueError(f"{split}: selected count must equal the declared minimum")
        for index in indices:
            contact = rows[index]
            candidate_id = str(contact["candidate_id"])
            item = source_by_candidate[candidate_id]
            if item["split"] != split or item["sample_id"] != contact["sample_id"]:
                raise ValueError(f"contact-sheet provenance mismatch: {split}/{index}")
            existing_eligible += int(bool(item["proposal_review"]["training_eligible"]))
            body = {
                "decision_id": f"uncertain-visual-v2:{candidate_id}",
                "candidate_id": candidate_id,
                "sample_id": item["sample_id"],
                "split": split,
                "source_group_id": item["source_group_id"],
                "effective_group_id": item["effective_group_id"],
                "decision": "UNCERTAIN_OR_OCCLUDED",
                "reason": ["belt_path_not_reliably_observable_in_classifier_roi"],
                "evidence_level": VISUAL_EVIDENCE_LEVEL,
                "reviewer_type": "MODEL_ASSISTED_VISUAL",
                "reviewer_model": VISUAL_REVIEWER_MODEL,
                "human_verified": False,
                "approval_status": "PENDING_HUMAN_APPROVAL",
                "governance_eligible": False,
            }
            body["decision_sha256"] = stable_json_hash(body)
            decisions.append(body)

    groups: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in ("source_group_id", "effective_group_id")
    }
    for item in decisions:
        for field in groups:
            groups[field][str(item[field])].add(str(item["split"]))
    group_report = {}
    for field, mapping in groups.items():
        leaks = sorted(group for group, splits in mapping.items() if len(splits) > 1)
        group_report[field] = {
            "unique_groups": len(mapping),
            "cross_split_leaks": leaks,
            "status": "PASS" if not leaks else "FAIL",
        }
    if any(report["status"] != "PASS" for report in group_report.values()):
        raise ValueError("uncertainty decisions contain cross-split group leakage")

    counts = Counter(item["split"] for item in decisions)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "review_method": "CONTACT_SHEET_VISUAL_SEMANTIC_REVIEW",
        "reviewer_type": "MODEL_ASSISTED_VISUAL",
        "reviewer_model": VISUAL_REVIEWER_MODEL,
        "source_review": str(source_review),
        "source_review_sha256": sha256_file(source_review),
        "contact_sheet_report": str(contact_root / "CONTACT_SHEET_REPORT.json"),
        "contact_sheet_report_sha256": sha256_file(
            contact_root / "CONTACT_SHEET_REPORT.json"
        ),
        "decision": "UNCERTAIN_OR_OCCLUDED",
        "evidence_definition": (
            "The classifier ROI does not expose a reliable seatbelt path because of crop, "
            "occlusion, severe darkness, blur, or contradictory visual evidence."
        ),
        "counts_by_split": {split: counts[split] for split in SPLITS},
        "minimum_counts_by_split": MIN_UNCERTAIN_COUNTS,
        "total_selected": len(decisions),
        "previously_training_eligible": existing_eligible,
        "newly_training_eligible": len(decisions) - existing_eligible,
        "group_isolation": group_report,
        "human_verified": False,
        "governance_eligible": False,
        "approval_status": "MODEL_ASSISTED_PENDING_APPROVAL",
        "policy": (
            "These decisions may open exploratory classifier training. They do not create "
            "HUMAN_APPROVED or governed ground truth."
        ),
        "decisions": decisions,
    }
    payload["decisions_sha256"] = hashlib.sha256(
        json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize model-assisted visual decisions for seatbelt uncertainty"
    )
    parser.add_argument(
        "--source-review",
        type=Path,
        default=Path(
            "datasets/manifests/pretrain_pending_approval/v2_seatbelt_uncertain.json"
        ),
    )
    parser.add_argument(
        "--contact-root",
        type=Path,
        default=Path("reports/diagnostics/seatbelt_uncertain_contact_sheets_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/manifests/pretrain_pending_approval/"
            "v2_seatbelt_uncertain_visual_decisions.json"
        ),
    )
    args = parser.parse_args()
    report = finalize(args.source_review, args.contact_root, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts_by_split": report["counts_by_split"],
                "newly_training_eligible": report["newly_training_eligible"],
                "group_isolation": report["group_isolation"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
