from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

CANONICAL_IDS = {"phone": 0, "seatbelt_fastened": 1, "seatbelt_unfastened": 2}


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def load_names(data_yaml: Path | None) -> dict[int, str]:
    if data_yaml is None:
        return {}
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def raw_yolo_labels(path: Path) -> list[tuple[int, list[float]]]:
    labels: list[tuple[int, list[float]]] = []
    if not path.exists():
        return labels
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid raw YOLO label {path}:{line_number}")
        class_id = int(parts[0])
        box = [float(value) for value in parts[1:]]
        if any(value < 0 or value > 1 for value in box) or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"invalid raw YOLO coordinates {path}:{line_number}")
        labels.append((class_id, box))
    return labels


def map_suggestions_from_label(
    label_path: Path,
    source_names: dict[int, str],
    mappings: dict,
) -> tuple[list[dict], list[dict]]:
    annotations: list[dict] = []
    rejected_suggestions: list[dict] = []
    try:
        source_labels = raw_yolo_labels(label_path)
    except ValueError as exc:
        return [], [{"source_class_id": None, "source_class_name": "INVALID_SOURCE_LABEL", "yolo": None, "reason": str(exc)}]
    for source_id, box in source_labels:
        source_name = source_names.get(source_id, f"UNKNOWN_CLASS_{source_id}")
        rule = mappings.get(normalize_name(source_name))
        if rule is None or rule.get("canonical") not in CANONICAL_IDS:
            rejected_suggestions.append(
                {
                    "source_class_id": source_id,
                    "source_class_name": source_name,
                    "yolo": box,
                    "reason": (rule or {}).get("reason", "No explicit canonical mapping"),
                }
            )
            continue
        annotations.append(
            {
                "class_id": CANONICAL_IDS[rule["canonical"]],
                "yolo": box,
                "occupant_role": "PENDING",
                "source_class_id": source_id,
                "source_class_name": source_name,
                "requires_review": True,
            }
        )
    return annotations, rejected_suggestions


def map_suggestions(
    image_path: Path,
    source_names: dict[int, str],
    mappings: dict,
) -> tuple[list[dict], list[dict]]:
    return map_suggestions_from_label(image_path.with_suffix(".txt"), source_names, mappings)


def build_queue(
    records: list[dict],
    source_names: dict[int, str],
    mappings: dict,
    source: dict,
) -> list[dict]:
    queue: list[dict] = []
    for record in records:
        if record.get("status") == "EXACT_DUPLICATE":
            continue
        image_path = Path(record["original_path"])
        label_path = Path(record["source_label_path"]) if record.get("source_label_path") else image_path.with_suffix(".txt")
        annotations, rejected = map_suggestions_from_label(label_path, source_names, mappings)
        queue.append(
            {
                "sample_id": record["sample_id"],
                "image_path": record["ingested_path"],
                "source_id": record["source_id"],
                "source_asset_id": record["source_asset_id"],
                "source_group_id": record["source_group_id"],
                "license": record["license"],
                "vehicle_context_required": True,
                "vehicle_context_id": None,
                "occupant_role": "PENDING",
                "review_tasks": source.get("review_tasks", []),
                "source_image_class_hint": record.get("source_image_class_hint"),
                "annotations": annotations,
                "excluded_source_annotations": rejected,
                "notes": "Confirm occupant is inside a vehicle; assign vehicle context and role evidence before approval.",
            }
        )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantics-safe human annotation queue")
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-data-yaml", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("datasets/sources.yaml"))
    parser.add_argument("--mappings", type=Path, default=Path("mappings/class_mapping.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/manifests/review_queue.json"))
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    source = next((item for item in registry["sources"] if item["source_id"] == args.source_id), None)
    if source is None:
        raise ValueError(f"unknown source_id: {args.source_id}")
    mapping_doc = yaml.safe_load(args.mappings.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.ingest_manifest.read_text(encoding="utf-8").splitlines() if line]
    records = [item for item in records if item.get("source_id") == args.source_id]
    queue = build_queue(records, load_names(args.source_data_yaml), mapping_doc["mappings"], source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"queue": str(args.output), "samples": len(queue)}, indent=2))


if __name__ == "__main__":
    main()
