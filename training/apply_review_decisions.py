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
    return f"{label.class_id} {label.x_center:.8f} {label.y_center:.8f} {label.width:.8f} {label.height:.8f}"


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
        }
        if status == "APPROVED":
            if vehicle_context_id is None:
                raise ValueError(f"approved sample {record['sample_id']} lacks vehicle_context_id")
            if occupant_role not in RESOLVED_OCCUPANT_ROLES:
                raise ValueError(f"approved sample {record['sample_id']} lacks a resolved occupant_role")
            annotations = decision.get("annotations", [])
            if not annotations:
                raise ValueError(f"approved sample {record['sample_id']} has no canonical annotations")
            unresolved = [
                index
                for index, annotation in enumerate(annotations)
                if annotation.get("occupant_role") not in RESOLVED_OCCUPANT_ROLES
            ]
            if unresolved:
                raise ValueError(
                    f"approved sample {record['sample_id']} has unresolved annotation roles: {unresolved}"
                )
            image_source = Path(record["ingested_path"])
            image_target = output / f"{record['sample_id']}{image_source.suffix.lower()}"
            label_target = output / f"{record['sample_id']}.txt"
            if image_target.exists() or label_target.exists():
                raise FileExistsError(f"reviewed output already exists for {record['sample_id']}")
            shutil.copy2(image_source, image_target)
            label_target.write_text("\n".join(yolo_line(annotation) for annotation in annotations) + "\n", encoding="utf-8")
            item.update(
                {
                    "reviewed_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "annotation_provenance": "human_reviewed",
                    "reviewed_annotations": annotations,
                    "occupant_roles": sorted({annotation["occupant_role"] for annotation in annotations}),
                    "effective_group_id": f"{record['source_group_id']}:{vehicle_context_id}",
                    "class_counts": dict(Counter(int(annotation["class_id"]) for annotation in annotations)),
                }
            )
        manifest.append(item)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize append-only human review decisions")
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument("--decisions", type=Path, default=Path("datasets/manifests/review_decisions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/reviewed"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/reviewed.jsonl"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.ingest_manifest.read_text(encoding="utf-8").splitlines() if line]
    result = apply(records, latest_decisions(args.decisions), args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(json.dumps(item, sort_keys=True) for item in result) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "decided_samples": len(result)}, indent=2))


if __name__ == "__main__":
    main()
