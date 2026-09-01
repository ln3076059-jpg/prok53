from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from training.pretrain_review_v2 import resolve_image

REVIEW_ROOT = Path("datasets/manifests/pretrain_pending_approval")
DRAFT = Path("datasets/manifests/human_review_drafts/v2_user_review_batch_001.json")


def _load(name: str) -> list[dict]:
    payload = json.loads((REVIEW_ROOT / name).read_text(encoding="utf-8"))
    return payload["samples"]


def _safe_name(sample_id: str, suffix: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in sample_id
    )
    return f"{safe}{suffix.lower()}"


def _link(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _export_group(root: Path, name: str, items: list[dict]) -> dict:
    target = root / name
    target.mkdir(parents=True, exist_ok=False)
    records = []
    modes = Counter()
    for item in items:
        source = resolve_image(item)
        destination = target / _safe_name(str(item["sample_id"]), source.suffix)
        mode = _link(source, destination)
        modes[mode] += 1
        review = item.get("proposal_review", {})
        records.append(
            {
                "sample_id": item["sample_id"],
                "image": destination.name,
                "source_image": str(source.resolve()),
                "source_group_id": item.get("source_group_id"),
                "proposal_decision": review.get("decision"),
                "proposal_tier": review.get("tier"),
                "proposal_confidence": review.get("confidence"),
                "reasons": review.get("reason", []),
                "training_eligible": review.get("training_eligible"),
            }
        )
    (target / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    return {"folder": str(target), "images": len(records), "materialization": dict(modes)}


def export(root: Path) -> dict:
    if root.exists():
        raise FileExistsError(f"review worklist already exists: {root}")
    root.mkdir(parents=True)
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    reviewed_ids = {item["sample_id"] for item in draft["decisions"]}

    phone_negative = _load("v2_phone_negative.json")
    phone_positive = _load("v2_phone_positive.json")
    seatbelt_roi = _load("v2_seatbelt_roi_state.json")
    seatbelt_uncertain = _load("v2_seatbelt_uncertain.json")
    adt = _load("v2_adt_external.json")
    mendeley = _load("v2_phone_external_mendeley.json")
    hard = _load("v2_hard_examples.json")
    by_id = {
        item["sample_id"]: item for group in (phone_negative, phone_positive) for item in group
    }

    groups = {
        "00_user_reviewed_pending_identity": [by_id[value] for value in sorted(reviewed_ids)],
        "01_phone_negative_model_conflict": [
            item
            for item in phone_negative
            if item["sample_id"] not in reviewed_ids
            and "phone_detector_found_candidate_in_proposed_negative"
            in item["proposal_review"]["reason"]
        ],
        "02_phone_negative_low_visibility": [
            item
            for item in phone_negative
            if item["sample_id"] not in reviewed_ids
            and "severe_visibility_or_tiny_object_risk" in item["proposal_review"]["reason"]
        ],
        "03_phone_positive_low_visibility": [
            item
            for item in phone_positive
            if item["sample_id"] not in reviewed_ids
            and item["proposal_review"]["tier"] == "UNCERTAIN"
        ],
        "04_seatbelt_roi_low_visibility": [
            item for item in seatbelt_roi if item["proposal_review"]["tier"] == "UNCERTAIN"
        ],
        "05_seatbelt_uncertain_classifier_candidates": seatbelt_uncertain,
        "06_external_adt_uncertain": [
            item for item in adt if item["proposal_review"]["tier"] == "UNCERTAIN"
        ],
        "07_external_mendeley_uncertain": [
            item for item in mendeley if item["proposal_review"]["tier"] == "UNCERTAIN"
        ],
        "08_hard_examples_uncertain": [
            item for item in hard if item["proposal_review"]["tier"] == "UNCERTAIN"
        ],
    }
    report = {name: _export_group(root, name, items) for name, items in groups.items()}
    total = sum(item["images"] for item in report.values())
    summary = {
        "schema_version": 1,
        "status": "EXPORTED",
        "total_images": total,
        "groups": report,
        "policy": (
            "Folders 01-05 are core review work. Folders 06-08 are external/hard-example "
            "follow-up and are not required for model-assisted pending-approval exploratory "
            "training. Folder 00 still needs reviewer identity and required metadata before "
            "governance admission."
        ),
    }
    (root / "WORKLIST_REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pending-approval images into review folders"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/review_worklists/v2_pending_approval"),
    )
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))


if __name__ == "__main__":
    main()
