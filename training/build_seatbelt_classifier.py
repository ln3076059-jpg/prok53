from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from training.common import stable_json_hash

CLASS_NAMES = {
    0: "seatbelt_fastened",
    1: "seatbelt_unfastened",
    2: "uncertain_or_occluded",
}
STATE_TO_CLASS = {value: key for key, value in CLASS_NAMES.items()}


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _find_image(root: Path, split: str, sample_id: str) -> Path:
    images = list((root / "images" / split).glob(f"{sample_id}.*"))
    if len(images) != 1:
        raise ValueError(f"expected one image for {split}/{sample_id}")
    return images[0]


def _crop(image, yolo: list[float], padding: float):
    height, width = image.shape[:2]
    x, y, box_width, box_height = yolo
    x1 = max(0, int((x - box_width / 2 - box_width * padding) * width))
    y1 = max(0, int((y - box_height / 2 - box_height * padding) * height))
    x2 = min(width, int((x + box_width / 2 + box_width * padding) * width))
    y2 = min(height, int((y + box_height / 2 + box_height * padding) * height))
    return image[y1:y2, x1:x2]


def build(
    source: Path,
    output: Path,
    padding: float = 0.08,
    uncertain_manifest: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not 0 <= padding <= 0.3:
        raise ValueError("padding must be between 0 and 0.3")
    manifest = _read_manifest(source / "manifest.jsonl")
    uncertain_rows = _read_manifest(uncertain_manifest) if uncertain_manifest else []
    uncertain_overrides = {
        (item.get("source_sample_id"), int(item.get("source_box_index", -1)))
        for item in uncertain_rows
        if item.get("source_sample_id") is not None
    }
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES.values():
            (output / split / class_name).mkdir(parents=True, exist_ok=False)

    crops = []
    counts = Counter()
    for item in manifest:
        image_path = _find_image(source, item["split"], item["sample_id"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode {image_path}")
        label_path = source / "labels" / item["split"] / f"{item['sample_id']}.txt"
        for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            if (item["sample_id"], box_index) in uncertain_overrides:
                continue
            parts = line.split()
            if int(parts[0]) != 0:
                raise ValueError(f"unexpected upper-body ROI class {parts[0]}")
            states = item.get("seatbelt_states", [])
            state = states[box_index] if box_index < len(states) else None
            if state not in STATE_TO_CLASS or state == "uncertain_or_occluded":
                raise ValueError(f"invalid source state for {item['sample_id']}: {state}")
            class_id = STATE_TO_CLASS[state]
            crop = _crop(image, [float(value) for value in parts[1:]], padding)
            if crop.size == 0 or min(crop.shape[:2]) < 12:
                raise ValueError(f"invalid crop {item['sample_id']} box {box_index}")
            crop_id = f"{item['sample_id']}_box_{box_index}"
            target = output / item["split"] / CLASS_NAMES[class_id] / f"{crop_id}.jpg"
            if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"could not write {target}")
            counts[(item["split"], class_id)] += 1
            crops.append(
                {
                    "crop_id": crop_id,
                    "sample_id": item["sample_id"],
                    "derived_from_sample_id": item.get("derived_from_sample_id"),
                    "split": item["split"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "source_group_id": item["source_group_id"],
                    "effective_group_id": item["effective_group_id"],
                    "source_image_sha256": item["sha256"],
                    "path": str(target.resolve()),
                }
            )
    for item in uncertain_rows:
        required = {"sample_id", "split", "image_path", "yolo", "source_group_id", "effective_group_id"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"uncertain record is missing fields: {sorted(missing)}")
        if item["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid uncertain split: {item['split']}")
        image_path = Path(item["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode uncertain image {image_path}")
        crop = _crop(image, [float(value) for value in item["yolo"]], padding)
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            raise ValueError(f"invalid uncertain crop: {item['sample_id']}")
        class_id = STATE_TO_CLASS["uncertain_or_occluded"]
        crop_id = f"{item['sample_id']}_uncertain"
        target = output / item["split"] / CLASS_NAMES[class_id] / f"{crop_id}.jpg"
        if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write {target}")
        counts[(item["split"], class_id)] += 1
        crops.append(
            {
                "crop_id": crop_id,
                "sample_id": item["sample_id"],
                "split": item["split"],
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "source_group_id": item["source_group_id"],
                "effective_group_id": item["effective_group_id"],
                "source_image_sha256": item.get("sha256"),
                "path": str(target.resolve()),
                "annotation_provenance": "HUMAN_REVIEWED_UNCERTAIN_OR_OCCLUDED",
            }
        )
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in crops) + "\n",
        encoding="utf-8",
    )
    for dimension in ("source_group_id", "effective_group_id"):
        split_by_group: defaultdict[str, set[str]] = defaultdict(set)
        for item in crops:
            split_by_group[str(item[dimension])].add(item["split"])
        overlaps = [group for group, splits in split_by_group.items() if len(splits) > 1]
        if overlaps:
            raise ValueError(f"classifier {dimension} crosses splits: {len(overlaps)}")
    report = {
        "task": "SEATBELT_STATE_CLASSIFICATION",
        "classes": CLASS_NAMES,
        "source_manifest_sha256": stable_json_hash(manifest),
        "crop_manifest_sha256": stable_json_hash(crops),
        "padding": padding,
        "counts": {
            split: {CLASS_NAMES[key]: counts[(split, key)] for key in CLASS_NAMES}
            for split in ("train", "val", "test")
        },
        "split_inheritance": "SOURCE_SPLITS_PRESERVED",
        "uncertain_manifest": str(uncertain_manifest.resolve()) if uncertain_manifest else None,
        "ready_for_training": all(
            counts[(split, class_id)] > 0
            for split in ("train", "val", "test")
            for class_id in CLASS_NAMES
        ),
        "uncertain_policy": "INVISIBLE_OR_OCCLUDED_IS_NOT_UNFASTENED",
        "reviewed_uncertain_overrides_replace_source_state": True,
        "human_review_status": "PENDING",
    }
    (output / "CLASSIFIER_DATASET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create belt-state classification crops")
    parser.add_argument(
        "--source", type=Path, default=Path("datasets/derived/seatbelt_v2_balanced")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/derived/seatbelt_classifier_v2")
    )
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument(
        "--uncertain-manifest",
        type=Path,
        help="Human-reviewed JSONL with uncertain/occluded upper-body crops and group-safe splits",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.padding, args.uncertain_manifest), indent=2))


if __name__ == "__main__":
    main()
