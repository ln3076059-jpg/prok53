from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from training.ingest_dataset import read_source_labels

SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "phone", 1: "seatbelt_fastened", 2: "seatbelt_unfastened"}
STATUS = "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA"


def clip_yolo_box(box: list[float], epsilon: float = 1e-9) -> tuple[list[float], bool]:
    center_x, center_y, width, height = box
    left = max(0.0, center_x - width / 2)
    right = min(1.0, center_x + width / 2)
    top = max(0.0, center_y - height / 2)
    bottom = min(1.0, center_y + height / 2)
    if right <= left or bottom <= top:
        raise ValueError(f"source box has no in-frame area: {box}")
    if left <= epsilon:
        left = epsilon
    if top <= epsilon:
        top = epsilon
    if right >= 1.0 - epsilon:
        right = 1.0 - epsilon
    if bottom >= 1.0 - epsilon:
        bottom = 1.0 - epsilon
    clipped = [
        (left + right) / 2,
        (top + bottom) / 2,
        right - left,
        bottom - top,
    ]
    changed = any(abs(left_value - right_value) > 1e-12 for left_value, right_value in zip(box, clipped))
    return clipped, changed


class UnionFind:
    def __init__(self, size: int):
        self.parents = list(range(size))
        self.sizes = [1] * size

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.sizes[left_root] < self.sizes[right_root]:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root
        self.sizes[left_root] += self.sizes[right_root]
        return True


class BKTree:
    def __init__(self):
        self.root: dict | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def query(self, value: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self.distance(value, node["value"])
            if distance <= threshold:
                matches.extend(node["indexes"])
            low, high = distance - threshold, distance + threshold
            pending.extend(
                child for edge, child in node["children"].items() if low <= edge <= high
            )
        return matches

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = {"value": value, "indexes": [index], "children": {}}
            return
        node = self.root
        while True:
            distance = self.distance(value, node["value"])
            if distance == 0:
                node["indexes"].append(index)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "value": value,
                    "indexes": [index],
                    "children": {},
                }
                return
            node = child


def multi_index_near_pairs(values: list[int], threshold: int) -> list[tuple[int, int]]:
    """Return every Hamming-near pair using threshold+1 exact-match partitions.

    If two 64-bit values differ in at most ``threshold`` bits, at least one of
    ``threshold + 1`` disjoint partitions has no differing bit. Bucket lookup
    therefore produces a complete candidate set without an O(n^2) scan.
    """
    if not 0 <= threshold < 64:
        raise ValueError("near threshold must be between 0 and 63")
    parts = threshold + 1
    buckets: list[dict[int, list[int]]] = [{} for _ in range(parts)]
    pairs: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        candidates: set[int] = set()
        signatures = [0] * parts
        # Interleave pHash bits across partitions to avoid low-entropy contiguous
        # buckets on grayscale imagery while keeping partitions disjoint.
        for bit in range(64):
            part = bit % parts
            signatures[part] |= ((value >> bit) & 1) << (bit // parts)
        for bucket, signature in zip(buckets, signatures, strict=True):
            candidates.update(bucket.get(signature, []))
        for other in candidates:
            if (value ^ values[other]).bit_count() <= threshold:
                pairs.append((other, index))
        for bucket, signature in zip(buckets, signatures, strict=True):
            bucket.setdefault(signature, []).append(index)
    return pairs


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def phone_records(dataset: Path) -> list[dict]:
    records = []
    for item in read_jsonl(dataset / "manifest.jsonl"):
        split = item["split"]
        image_candidates = list((dataset / "images" / split).glob(f"{item['sample_id']}.*"))
        if len(image_candidates) != 1:
            raise ValueError(f"expected one phone image for {item['sample_id']}")
        label_path = dataset / "labels" / split / f"{item['sample_id']}.txt"
        labels = []
        clipped_boxes = 0
        for class_id, box in read_source_labels(label_path):
            clipped, changed = clip_yolo_box(box)
            labels.append((class_id, clipped))
            clipped_boxes += changed
        if any(class_id != 0 for class_id, _ in labels):
            raise ValueError("phone bootstrap contains a non-phone label")
        records.append(
            {
                "sample_id": item["sample_id"],
                "source_id": item["source_id"],
                "source_asset_id": item["source_asset_id"],
                "source_group_id": item["source_group_id"],
                "base_group_id": f"phone:{item['effective_group_id']}",
                "image_path": image_candidates[0],
                "labels": labels,
                "sha256": item["sha256"],
                "phash": item["phash"],
                "clipped_boxes": clipped_boxes,
            }
        )
    return records


def seatbelt_records(manifest: Path) -> list[dict]:
    records = []
    for item in read_jsonl(manifest):
        if item.get("source_id") != "roboflow_seatbelttraining_v4" or item.get("status") == "EXACT_DUPLICATE":
            continue
        source_labels = read_source_labels(Path(item["source_label_path"]))
        labels = []
        clipped_boxes = 0
        for source_class_id, box in source_labels:
            if source_class_id not in {0, 1}:
                raise ValueError(f"unexpected Roboflow seatbelt class: {source_class_id}")
            clipped, changed = clip_yolo_box(box)
            labels.append((2 if source_class_id == 0 else 1, clipped))
            clipped_boxes += changed
        if not labels:
            raise ValueError(f"seatbelt sample has no state box: {item['sample_id']}")
        records.append(
            {
                "sample_id": item["sample_id"],
                "source_id": item["source_id"],
                "source_asset_id": item["source_asset_id"],
                "source_group_id": item["source_group_id"],
                "base_group_id": f"seatbelt:{item['source_group_id']}",
                "image_path": Path(item["original_path"]),
                "labels": labels,
                "sha256": item["sha256"],
                "phash": item["phash"],
                "clipped_boxes": clipped_boxes,
            }
        )
    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    seen: set[str] = set()
    for item in records:
        if item["sha256"] in seen:
            continue
        seen.add(item["sha256"])
        kept.append(item)
    return kept, len(records) - len(kept)


def cluster(records: list[dict], near_threshold: int) -> tuple[dict[int, str], dict]:
    union_find = UnionFind(len(records))
    first_by_group: dict[str, int] = {}
    first_by_sha: dict[str, int] = {}
    for index, item in enumerate(records):
        if item["base_group_id"] in first_by_group:
            union_find.union(index, first_by_group[item["base_group_id"]])
        else:
            first_by_group[item["base_group_id"]] = index
        if item["sha256"] in first_by_sha:
            union_find.union(index, first_by_sha[item["sha256"]])
        else:
            first_by_sha[item["sha256"]] = index

    values = [int(item["phash"], 16) for item in records]
    near_matches = multi_index_near_pairs(values, near_threshold)
    for left, right in near_matches:
        union_find.union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(union_find.find(index), []).append(index)
    component_ids = {}
    for indexes in components.values():
        component_id = "near:" + hashlib.sha256(
            "\n".join(sorted(records[index]["sample_id"] for index in indexes)).encode()
        ).hexdigest()[:20]
        for index in indexes:
            component_ids[index] = component_id
    return component_ids, {
        "near_threshold": near_threshold,
        "near_pairs": len(near_matches),
        "components": len(components),
        "largest_component": max(map(len, components.values()), default=0),
        "component_indexes": list(components.values()),
    }


def assign_splits(records: list[dict], clustering: dict) -> dict[int, str]:
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    total_images = len(records)
    total_classes = Counter(
        class_id for item in records for class_id, _ in item["labels"]
    )
    images = Counter()
    classes = {split: Counter() for split in SPLITS}
    assignments = {}

    components = clustering["component_indexes"]
    components.sort(
        key=lambda indexes: (
            len(indexes),
            sum(len(records[index]["labels"]) for index in indexes),
        ),
        reverse=True,
    )
    for indexes in components:
        component_classes = Counter(
            class_id for index in indexes for class_id, _ in records[index]["labels"]
        )

        def deficit(split: str) -> float:
            score = (targets[split] * total_images - images[split]) / max(total_images, 1)
            score += sum(
                (targets[split] * total_classes[class_id] - classes[split][class_id])
                / max(total_classes[class_id], 1)
                for class_id in CLASS_NAMES
            )
            return score

        split = max(SPLITS, key=deficit)
        for index in indexes:
            assignments[index] = split
        images[split] += len(indexes)
        classes[split].update(component_classes)
    return assignments


def build(phone_dataset: Path, seatbelt_manifest: Path, output: Path, near_threshold: int) -> dict:
    if output.exists():
        raise FileExistsError(f"bootstrap output already exists: {output}")
    records, exact_duplicates_skipped = deduplicate(
        [*phone_records(phone_dataset), *seatbelt_records(seatbelt_manifest)]
    )
    component_ids, clustering = cluster(records, near_threshold)
    assignments = assign_splits(records, clustering)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    manifest = []
    image_counts = Counter()
    class_counts = {split: Counter() for split in SPLITS}
    positive_images = {split: Counter() for split in SPLITS}
    source_counts = {split: Counter() for split in SPLITS}
    split_components = {split: set() for split in SPLITS}
    for index, item in enumerate(records):
        split = assignments[index]
        suffix = item["image_path"].suffix.lower()
        image_target = output / "images" / split / f"{item['sample_id']}{suffix}"
        label_target = output / "labels" / split / f"{item['sample_id']}.txt"
        shutil.copy2(item["image_path"], image_target)
        label_target.write_text(
            "\n".join(
                f"{class_id} " + " ".join(f"{value:.16g}" for value in box)
                for class_id, box in item["labels"]
            )
            + ("\n" if item["labels"] else ""),
            encoding="utf-8",
        )
        counts = Counter(class_id for class_id, _ in item["labels"])
        image_counts[split] += 1
        class_counts[split].update(counts)
        positive_images[split].update(counts.keys())
        source_counts[split][item["source_id"]] += 1
        split_components[split].add(component_ids[index])
        manifest.append(
            {
                "sample_id": item["sample_id"],
                "source_id": item["source_id"],
                "source_asset_id": item["source_asset_id"],
                "source_group_id": item["source_group_id"],
                "base_group_id": item["base_group_id"],
                "effective_group_id": component_ids[index],
                "near_cluster_id": component_ids[index],
                "split": split,
                "class_counts": {str(key): value for key, value in sorted(counts.items())},
                "usage": "PROPOSAL_MODEL_ONLY",
                "human_review_status": "PENDING",
                "vehicle_context_required": True,
                "sha256": item["sha256"],
                "phash": item["phash"],
                "clipped_boxes": item["clipped_boxes"],
            }
        )

    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 3,
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n",
        encoding="utf-8",
    )
    overlap = {
        "train_val": len(split_components["train"] & split_components["val"]),
        "train_test": len(split_components["train"] & split_components["test"]),
        "val_test": len(split_components["val"] & split_components["test"]),
    }
    summary = {
        "status": STATUS,
        "classes": CLASS_NAMES,
        "images": {split: image_counts[split] for split in SPLITS},
        "class_instances": {
            split: {CLASS_NAMES[key]: class_counts[split][key] for key in CLASS_NAMES}
            for split in SPLITS
        },
        "positive_images": {
            split: {CLASS_NAMES[key]: positive_images[split][key] for key in CLASS_NAMES}
            for split in SPLITS
        },
        "source_images": {split: dict(source_counts[split]) for split in SPLITS},
        "exact_duplicates_skipped": exact_duplicates_skipped,
        "source_boxes_clipped_to_image": sum(item["clipped_boxes"] for item in records),
        "component_overlap": overlap,
        "near_duplicate_clustering": {
            key: value for key, value in clustering.items() if key != "component_indexes"
        },
        "policy": "Bootstrap/proposal training only. Human review remains mandatory before MC_001 final training or accuracy claims.",
    }
    (output / "BOOTSTRAP_MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe three-class proposal bootstrap")
    parser.add_argument(
        "--phone-dataset", type=Path, default=Path("datasets/derived/phone_bootstrap_v2")
    )
    parser.add_argument(
        "--seatbelt-manifest",
        type=Path,
        default=Path("datasets/manifests/ingest_roboflow_seatbelttraining_v4.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/derived/mc_bootstrap_v1")
    )
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.phone_dataset, args.seatbelt_manifest, args.output, args.near_threshold),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
