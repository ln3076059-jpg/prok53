from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common import sha256_file
from training.run_review1_review import _load_queue, _resolve_image


def record(
    queue_path: Path,
    contact_sheet_dir: Path,
    judgments_path: Path,
    output: Path,
    datasets_root: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    queue = {str(item["sample_id"]): item for item in _load_queue(queue_path)}
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    accepted = {int(value) for value in judgments["accepted_indexes"]}
    rejected = {int(value) for value in judgments["rejected_indexes"]}
    if accepted & rejected:
        raise ValueError("accepted and rejected judgments overlap")
    page_rows = {}
    page_evidence = {}
    for manifest in sorted(contact_sheet_dir.glob("page_*.json")):
        image = manifest.with_suffix(".jpg")
        if not image.is_file():
            raise FileNotFoundError(image)
        page_evidence[manifest.name] = {
            "contact_sheet_path": str(image),
            "contact_sheet_sha256": sha256_file(image),
        }
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            index = int(item["index"])
            if index in page_rows:
                raise ValueError(f"duplicate contact-sheet index: {index}")
            page_rows[index] = {**item, "page_manifest": manifest.name}
    judged = accepted | rejected
    if judged != set(page_rows):
        missing = sorted(set(page_rows) - judged)
        unknown = sorted(judged - set(page_rows))
        raise ValueError(
            "judgments must cover every displayed sample; "
            f"missing={missing}, unknown={unknown}"
        )

    records = []
    for index in sorted(judged):
        sheet_item = page_rows[index]
        sample_id = str(sheet_item["sample_id"])
        item = queue[sample_id]
        image = _resolve_image(item, datasets_root)
        evidence = {
            "inspected": True,
            "method": "CONTACT_SHEET_IMAGE_INSPECTION",
            "contact_sheet_index": index,
            **page_evidence[sheet_item["page_manifest"]],
        }
        if index in accepted:
            record_item = {
                "sample_id": sample_id,
                "source_sha256": sha256_file(image),
                "status": "REVIEW1_ACCEPTED_PROPOSAL",
                "decision_reason": "HARD_NEGATIVE_CONFIRMED",
                "reviewed_annotations": item.get("annotations", []),
                "review1_confidence": 0.94,
                "risk_flags": [],
                "notes": (
                    "Visible in-vehicle occupant; no physical phone visible in the full frame."
                ),
                "visual_evidence": evidence,
                "vehicle_context_id": "UNKNOWN",
                "occupant_role": "driver",
                "video_id": item.get("proposal_review", {})
                .get("metadata_suggestions", {})
                .get("video_id", "UNKNOWN"),
                "vehicle_id": "UNKNOWN",
                "person_id": "UNKNOWN",
                "camera_id": "UNKNOWN",
                "conditions": [],
            }
        else:
            record_item = {
                "sample_id": sample_id,
                "source_sha256": sha256_file(image),
                "status": "REVIEW1_REJECTED_PROPOSAL",
                "decision_reason": "OTHER_DOCUMENTED_IN_NOTES",
                "reviewed_annotations": [],
                "review1_confidence": 0.99,
                "risk_flags": ["OUT_OF_VEHICLE_OR_INVALID_OCCUPANT_CONTEXT"],
                "notes": (
                    "Rejected from the driver-cabin phone-negative lane: the visible scene is "
                    "outside a valid in-vehicle occupant context or contains no usable occupant."
                ),
                "visual_evidence": evidence,
                "vehicle_context_id": "UNKNOWN",
                "occupant_role": "UNCERTAIN",
                "video_id": "UNKNOWN",
                "vehicle_id": "UNKNOWN",
                "person_id": "UNKNOWN",
                "camera_id": "UNKNOWN",
                "conditions": [],
            }
        records.append(record_item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "reviewed": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "judgments_sha256": sha256_file(judgments_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record explicit Review 1 contact-sheet judgments")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    print(
        json.dumps(
            record(
                args.queue,
                args.contact_sheet_dir,
                args.judgments,
                args.output,
                args.datasets_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
