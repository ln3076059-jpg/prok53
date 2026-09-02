from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from training.common import YoloLabel

RESOLVED_OCCUPANT_ROLES = {
    "driver",
    "front_passenger",
    "rear_left",
    "rear_center",
    "rear_right",
    "other_occupant",
}
REQUIRED_REVIEW_METADATA = ("video_id", "vehicle_id", "person_id", "camera_id")


def latest_decisions(path: Path) -> dict[str, dict]:
    decisions: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            decisions[item["sample_id"]] = item
    return decisions


def yolo_line(annotation: dict) -> str:
    label = YoloLabel(int(annotation["class_id"]), *(float(value) for value in annotation["yolo"]))
    label.validate()
    return (
        f"{label.class_id} {label.x_center:.8f} {label.y_center:.8f} "
        f"{label.width:.8f} {label.height:.8f}"
    )


def resolve_ingested_image(record: dict, incoming_root: Path = Path("datasets/incoming")) -> Path:
    image = Path(record["ingested_path"])
    if image.is_file():
        return image
    parts = image.parts
    incoming_index = next(
        (index for index, value in enumerate(parts) if value.lower() == "incoming"),
        None,
    )
    if incoming_index is not None:
        relocated = incoming_root.joinpath(*parts[incoming_index + 1 :])
        if relocated.is_file():
            return relocated
    sample_id = record["sample_id"]
    matches = [path for path in incoming_root.rglob(f"{sample_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"stale ingested_path for {sample_id}; expected one fallback image, found "
            f"{len(matches)}"
        )
    return matches[0]


def apply(records: list[dict], decisions: dict[str, dict], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for record in records:
        decision = decisions.get(record.get("sample_id"))
        if decision is None or record.get("status") == "EXACT_DUPLICATE":
            continue
        status = decision["status"]
        vehicle_context_id = (decision.get("vehicle_context_id") or "").strip() or None
        occupant_role = decision.get("occupant_role")
        item = {
            **record,
            "review_status": status,
            "review_notes": decision.get("notes", ""),
            "reviewed_at_utc": decision.get("reviewed_at_utc"),
            "vehicle_context_id": vehicle_context_id,
            "occupant_role": occupant_role,
            "reviewer_id": decision.get("reviewer_id"),
            "reviewer_type": decision.get("reviewer_type"),
            "decision_id": decision.get("decision_id"),
            "previous_status": decision.get("previous_status"),
            "new_status": decision.get("new_status", status),
            "decision_reason": decision.get("decision_reason"),
            "decision_source": decision.get("source"),
            "video_id": decision.get("video_id"),
            "vehicle_id": decision.get("vehicle_id"),
            "person_id": decision.get("person_id"),
            "camera_id": decision.get("camera_id"),
            "conditions": decision.get("conditions", []),
            "proposal_review_snapshot": decision.get("proposal_review_snapshot"),
        }
        if status in {"APPROVED", "APPROVED_NEGATIVE"}:
            if not decision.get("reviewer_id") or decision.get("reviewer_type") != "HUMAN":
                raise ValueError(
                    f"approved sample {record['sample_id']} lacks named human reviewer provenance"
                )
            missing_metadata = [
                field for field in REQUIRED_REVIEW_METADATA if not decision.get(field)
            ]
            if missing_metadata:
                raise ValueError(
                    f"approved sample {record['sample_id']} lacks metadata: {missing_metadata}"
                )
            if not decision.get("conditions"):
                raise ValueError(f"approved sample {record['sample_id']} lacks conditions")
            if vehicle_context_id is None:
                raise ValueError(f"approved sample {record['sample_id']} lacks vehicle_context_id")
            if occupant_role not in RESOLVED_OCCUPANT_ROLES:
                raise ValueError(
                    f"approved sample {record['sample_id']} lacks a resolved occupant_role"
                )
            annotations = decision.get("annotations", [])
            if status == "APPROVED" and not annotations:
                raise ValueError(
                    f"approved sample {record['sample_id']} has no canonical annotations"
                )
            if status == "APPROVED_NEGATIVE" and annotations:
                raise ValueError(
                    f"approved negative sample {record['sample_id']} contains annotations"
                )
            unresolved = [
                index
                for index, annotation in enumerate(annotations)
                if annotation.get("occupant_role") not in RESOLVED_OCCUPANT_ROLES
            ]
            if unresolved:
                raise ValueError(
                    f"approved sample {record['sample_id']} has unresolved annotation roles: "
                    f"{unresolved}"
                )
            image_source = resolve_ingested_image(record)
            image_target = output / f"{record['sample_id']}{image_source.suffix.lower()}"
            label_target = output / f"{record['sample_id']}.txt"
            if image_target.exists() or label_target.exists():
                raise FileExistsError(f"reviewed output already exists for {record['sample_id']}")
            shutil.copy2(image_source, image_target)
            label_text = "\n".join(yolo_line(annotation) for annotation in annotations)
            label_target.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
            item.update(
                {
                    "reviewed_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "annotation_provenance": (
                        "human_reviewed_negative"
                        if status == "APPROVED_NEGATIVE"
                        else "human_reviewed"
                    ),
                    "reviewed_annotations": annotations,
                    "occupant_roles": sorted(
                        {annotation["occupant_role"] for annotation in annotations}
                    ),
                    "effective_group_id": f"{record['source_group_id']}:{vehicle_context_id}",
                    "class_counts": dict(
                        Counter(int(annotation["class_id"]) for annotation in annotations)
                    ),
                }
            )
        manifest.append(item)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize append-only human review decisions")
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument(
        "--decisions", type=Path, default=Path("datasets/manifests/review_decisions.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/reviewed"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/reviewed.jsonl"))
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.ingest_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result = apply(records, latest_decisions(args.decisions), args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in result) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(args.manifest), "decided_samples": len(result)}, indent=2))


if __name__ == "__main__":
    main()
