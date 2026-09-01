from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from training.common import perceptual_hash, sha256_file, stable_json_hash

SOURCE_ID = "roboflow_adt_seatbelt_v1"
LICENSE = "CC-BY-4.0"
CANONICAL_PROPOSALS = {
    7: (0, "Candidate physical phone; confirm object and occupant context."),
    9: (
        2,
        (
            "Candidate unfastened occupant; require visible evidence and redraw "
            "comparable upper-body ROI."
        ),
    ),
    10: (
        1,
        "Candidate fastened occupant; verify visible belt and redraw comparable upper-body ROI.",
    ),
}
RF_SUFFIX = re.compile(r"^(?P<base>.+?)(?:_jpg)?\.rf\.[0-9a-f]+$", re.IGNORECASE)


def _source_group(stem: str) -> str:
    match = RF_SUFFIX.match(stem)
    base = match.group("base") if match else stem
    return f"adt:{base}"


def _parse_label(path: Path, class_count: int) -> tuple[list[dict], list[str]]:
    output = []
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"malformed source label {path}:{line_number}")
        class_id = int(parts[0])
        yolo = [float(value) for value in parts[1:]]
        x, y, width, height = yolo
        if not all(math.isfinite(value) for value in yolo):
            raise ValueError(f"non-finite source box {path}:{line_number}")
        if not 0 <= class_id < class_count:
            raise ValueError(f"source class out of range {path}:{line_number}")
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive source box {path}:{line_number}")
        x1 = max(0.0, x - width / 2)
        y1 = max(0.0, y - height / 2)
        x2 = min(1.0, x + width / 2)
        y2 = min(1.0, y + height / 2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"source box has no in-frame area {path}:{line_number}")
        clipped = [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ]
        was_clipped = any(
            abs(left - right) > 1e-9 for left, right in zip(yolo, clipped, strict=True)
        )
        if was_clipped:
            errors.append(f"line {line_number}: source box clipped to image bounds")
        output.append(
            {
                "class_id": class_id,
                "source_yolo": yolo,
                "yolo": clipped,
                "source_box_clipped": was_clipped,
            }
        )
    return output, errors


def build(source: Path, queue_output: Path, ingest_output: Path, report_output: Path) -> dict:
    data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = [str(value) for value in data["names"]]
    images = sorted((source / "train" / "images").glob("*"))
    if not images:
        raise ValueError(f"no ADT images found under {source}")
    records = []
    queue_candidates = []
    class_boxes: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    by_group: defaultdict[str, list[dict]] = defaultdict(list)
    for image_path in images:
        if not image_path.is_file():
            continue
        label_path = source / "train" / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing ADT label: {label_path}")
        with Image.open(image_path) as image:
            image.verify()
        source_labels, source_label_errors = _parse_label(label_path, len(names))
        present_classes = set()
        annotations = []
        excluded = []
        for source_label in source_labels:
            class_id = int(source_label["class_id"])
            yolo = source_label["yolo"]
            class_name = names[class_id]
            class_boxes[class_name] += 1
            present_classes.add(class_name)
            proposal = CANONICAL_PROPOSALS.get(class_id)
            if proposal:
                canonical_id, reason = proposal
                annotations.append(
                    {
                        "class_id": canonical_id,
                        "occupant_role": "PENDING",
                        "requires_review": True,
                        "source_class_id": class_id,
                        "source_class_name": class_name,
                        "proposal_reason": reason,
                        "source_box_clipped": source_label["source_box_clipped"],
                        "source_yolo": source_label["source_yolo"],
                        "yolo": yolo,
                    }
                )
            else:
                excluded.append(
                    {
                        "source_class_id": class_id,
                        "source_class_name": class_name,
                        "reason": "Unrelated/inconsistent ADT class; never map automatically.",
                        "source_box_clipped": source_label["source_box_clipped"],
                        "source_yolo": source_label["source_yolo"],
                        "yolo": yolo,
                    }
                )
        class_images.update(present_classes)
        digest = sha256_file(image_path)
        sample_id = digest[:32]
        group = _source_group(image_path.stem)
        record = {
            "annotation_count": len(source_labels),
            "annotation_provenance": "source_import_requires_human_review",
            "ingested_path": str(image_path.resolve()),
            "license": LICENSE,
            "original_path": str(image_path.resolve()),
            "phash": perceptual_hash(image_path),
            "review_status": "PENDING",
            "sample_id": sample_id,
            "sha256": digest,
            "source_asset_id": str(image_path.relative_to(source)).replace("\\", "/"),
            "source_group_id": group,
            "source_group_status": "INFERRED_ROBOFLOW_AUGMENTATION_GROUP",
            "source_id": SOURCE_ID,
            "source_label_path": str(label_path.resolve()),
            "source_label_errors": source_label_errors,
            "vehicle_context_id": None,
            "vehicle_context_required": True,
        }
        queue_item = {
            "annotations": annotations,
            "excluded_source_annotations": excluded,
            "image_path": str(image_path.resolve()),
            "license": LICENSE,
            "intended_use": "SOURCE_DISJOINT_EXTERNAL_TEST_CANDIDATE",
            "split_policy": "EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK",
            "notes": (
                "ADT is a mixed 12-class proposal source. Verify physical phone or visible belt "
                "state, redraw a comparable upper-body ROI, and resolve occupant/vehicle context."
            ),
            "occupant_role": "PENDING",
            "review_tasks": [
                "CLASS_SCHEMA_REVIEW",
                "SUBJECT_VEHICLE_CAMERA_GROUPING",
                "UPPER_BODY_REBOX",
                "SEATBELT_STATE_REVIEW",
                "HARD_NEGATIVE_REVIEW",
            ],
            "sample_id": sample_id,
            "source_asset_id": record["source_asset_id"],
            "source_group_id": group,
            "source_id": SOURCE_ID,
            "source_image_class_hint": sorted(present_classes),
            "source_label_errors": source_label_errors,
            "vehicle_context_id": None,
            "vehicle_context_required": True,
        }
        records.append(record)
        queue_candidates.append(queue_item)
        by_group[group].append(queue_item)

    selected = []
    for _group, variants in sorted(by_group.items()):
        selected.append(
            sorted(
                variants,
                key=lambda item: (
                    -len(item["annotations"]),
                    len(item["excluded_source_annotations"]),
                    item["sample_id"],
                ),
            )[0]
        )
    selected.sort(key=lambda item: (-len(item["annotations"]), item["source_group_id"]))
    record_by_id = {item["sample_id"]: item for item in records}
    selected_records = [record_by_id[item["sample_id"]] for item in selected]
    queue_output.parent.mkdir(parents=True, exist_ok=True)
    queue_output.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    ingest_output.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in selected_records) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "HUMAN_REVIEW_REQUIRED",
        "source_id": SOURCE_ID,
        "license": LICENSE,
        "source_images": len(records),
        "inferred_groups": len(by_group),
        "review_representatives": len(selected),
        "representatives_with_canonical_proposals": sum(
            bool(item["annotations"]) for item in selected
        ),
        "class_boxes": dict(sorted(class_boxes.items())),
        "class_images": dict(sorted(class_images.items())),
        "queue_sha256": sha256_file(queue_output),
        "ingest_manifest_sha256": sha256_file(ingest_output),
        "selection_policy": "ONE_REPRESENTATIVE_PER_INFERRED_ROBOFLOW_AUGMENTATION_GROUP",
        "automatic_training_admission": False,
        "governed_split_policy": "EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK",
        "images_with_source_label_errors": sum(
            bool(item.get("source_label_errors")) for item in selected_records
        ),
        "content_fingerprint": stable_json_hash(selected),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the governed ADT external-source review queue"
    )
    parser.add_argument(
        "--source", type=Path, default=Path("datasets/raw/roboflow_adt_seatbelt_v1/1")
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("datasets/manifests/review_queue_adt_seatbelt_v1.json"),
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        default=Path("datasets/manifests/ingest_adt_seatbelt_v1.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/adt_seatbelt_v1/review_source_audit.json")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.queue, args.ingest, args.report), indent=2))


if __name__ == "__main__":
    main()
