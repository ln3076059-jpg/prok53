from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from training.common import hamming_distance
from training.ingest_dataset import read_source_labels, source_group_id


def split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def load_records(path: Path, source_id: str) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("source_id") == source_id and record.get("status") != "EXACT_DUPLICATE":
            records.append(record)
    return records


def prepare_records(records: list[dict], group_prefix: str, phone_source_class_id: int) -> list[dict]:
    prepared = []
    for record in records:
        image_path = Path(record["original_path"])
        label_path = Path(record["source_label_path"]) if record.get("source_label_path") else None
        source_labels = read_source_labels(label_path) if label_path else []
        prepared.append(
            {
                "record": record,
                "image_path": image_path,
                "phone_boxes_data": [
                    box for class_id, box in source_labels if class_id == phone_source_class_id
                ],
                "source_group_id": source_group_id(image_path, group_prefix),
            }
        )
    return prepared


def leakage_safe_assignments(prepared: list[dict], near_threshold: int) -> tuple[dict[int, str], dict]:
    parents = list(range(len(prepared)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        return True

    first_by_group: dict[str, int] = {}
    for index, item in enumerate(prepared):
        group_id = item["source_group_id"]
        if group_id in first_by_group:
            union(index, first_by_group[group_id])
        else:
            first_by_group[group_id] = index

    hashes = [item["record"].get("phash") for item in prepared]
    near_pairs = 0
    for left in range(len(hashes)):
        if not hashes[left]:
            continue
        for right in range(left + 1, len(hashes)):
            if hashes[right] and hamming_distance(hashes[left], hashes[right]) <= near_threshold:
                near_pairs += 1
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(prepared)):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(
        components.values(),
        key=lambda indexes: (len(indexes), sum(len(prepared[index]["phone_boxes_data"]) for index in indexes)),
        reverse=True,
    )
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    total_images = len(prepared)
    total_boxes = sum(len(item["phone_boxes_data"]) for item in prepared)
    image_counts = Counter()
    box_counts = Counter()
    assignments: dict[int, str] = {}
    component_ids: dict[int, str] = {}
    for indexes in ordered:
        component_boxes = sum(len(prepared[index]["phone_boxes_data"]) for index in indexes)

        def deficit(split: str) -> float:
            image_deficit = (targets[split] * total_images - image_counts[split]) / max(total_images, 1)
            box_deficit = (targets[split] * total_boxes - box_counts[split]) / max(total_boxes, 1)
            return image_deficit + box_deficit

        split = max(targets, key=deficit)
        component_id = "near:" + hashlib.sha256(
            "\n".join(sorted(prepared[index]["record"]["sample_id"] for index in indexes)).encode()
        ).hexdigest()[:20]
        for index in indexes:
            assignments[index] = split
            component_ids[index] = component_id
        image_counts[split] += len(indexes)
        box_counts[split] += component_boxes
    return assignments, {
        "component_ids": component_ids,
        "near_threshold": near_threshold,
        "near_pairs": near_pairs,
        "components": len(components),
        "largest_component": max((len(indexes) for indexes in components.values()), default=0),
    }


def build(
    records: list[dict],
    output: Path,
    group_prefix: str,
    phone_source_class_id: int,
    near_threshold: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"bootstrap output already exists: {output}")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    prepared = prepare_records(records, group_prefix, phone_source_class_id)
    assignments, cluster_metadata = leakage_safe_assignments(prepared, near_threshold)
    manifest: list[dict] = []
    split_images = Counter()
    split_positive_images = Counter()
    split_phone_boxes = Counter()
    split_groups: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    for index, item in enumerate(prepared):
        record = item["record"]
        image_path = item["image_path"]
        phone_boxes = item["phone_boxes_data"]
        group_id = item["source_group_id"]
        split = assignments[index]
        image_target = output / "images" / split / f"{record['sample_id']}{image_path.suffix.lower()}"
        label_target = output / "labels" / split / f"{record['sample_id']}.txt"
        shutil.copy2(image_path, image_target)
        label_target.write_text(
            "\n".join("0 " + " ".join(f"{value:.16g}" for value in box) for box in phone_boxes)
            + ("\n" if phone_boxes else ""),
            encoding="utf-8",
        )
        split_images[split] += 1
        split_phone_boxes[split] += len(phone_boxes)
        split_positive_images[split] += bool(phone_boxes)
        split_groups[split].add(group_id)
        manifest.append(
            {
                "sample_id": record["sample_id"],
                "source_id": record["source_id"],
                "source_asset_id": record["source_asset_id"],
                "source_group_id": group_id,
                "effective_group_id": cluster_metadata["component_ids"][index],
                "near_cluster_id": cluster_metadata["component_ids"][index],
                "split": split,
                "phone_boxes": len(phone_boxes),
                "usage": "PROPOSAL_MODEL_ONLY",
                "human_review_status": "PENDING",
                "vehicle_context_required": True,
                "sha256": record.get("sha256"),
                "phash": record.get("phash"),
            }
        )

    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "phone"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA",
        "images": dict(split_images),
        "positive_images": dict(split_positive_images),
        "phone_boxes": dict(split_phone_boxes),
        "groups": {split: len(groups) for split, groups in split_groups.items()},
        "group_overlap": {
            "train_val": len(split_groups["train"] & split_groups["val"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "val_test": len(split_groups["val"] & split_groups["test"]),
        },
        "near_duplicate_clustering": {
            key: value for key, value in cluster_metadata.items() if key != "component_ids"
        },
        "policy": "Use weights only to generate human-review proposals. Do not report these labels as final ground truth.",
    }
    (output / "BOOTSTRAP_MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe phone proposal bootstrap dataset")
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument("--source-id", default="kaggle_habbas_dms_v1")
    parser.add_argument("--group-prefix", default="dms-phone-bootstrap-v2")
    parser.add_argument("--phone-source-class-id", type=int, default=3)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("datasets/derived/phone_bootstrap_v2"))
    args = parser.parse_args()
    summary = build(
        load_records(args.ingest_manifest, args.source_id),
        args.output,
        args.group_prefix,
        args.phone_source_class_id,
        args.near_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
