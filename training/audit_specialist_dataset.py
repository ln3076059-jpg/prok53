from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from training.common import parse_yolo_line, sha256_file, stable_json_hash

EXPECTED_NAMES = {0: "occupant_upper_body"}
SPLITS = ("train", "val", "test")


def audit(root: Path) -> dict:
    errors: list[str] = []
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = {int(key): value for key, value in data.get("names", {}).items()}
    if names != EXPECTED_NAMES or int(data.get("nc", -1)) != len(EXPECTED_NAMES):
        errors.append(f"class schema mismatch: {names}")
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_id = {item["sample_id"]: item for item in manifest}
    if len(by_id) != len(manifest):
        errors.append("duplicate sample_id in manifest")
    split_by_dimension: dict[str, defaultdict[str, set[str]]] = {
        name: defaultdict(set)
        for name in ("sha256", "source_group_id", "effective_group_id", "near_cluster_id")
    }
    for item in manifest:
        for name, values in split_by_dimension.items():
            values[str(item[name])].add(item["split"])
        if item.get("derived_from_sample_id") and item["split"] != "train":
            errors.append(f"augmentation outside train: {item['sample_id']}")
    overlap = {
        name: sorted(key for key, splits in values.items() if len(splits) > 1)
        for name, values in split_by_dimension.items()
    }
    for name, values in overlap.items():
        if values:
            errors.append(f"{name} crosses splits: {len(values)}")

    split_report = {}
    for split in SPLITS:
        images = [path for path in (root / "images" / split).glob("*") if path.is_file()]
        labels = [path for path in (root / "labels" / split).glob("*.txt") if path.is_file()]
        expected = sum(item["split"] == split for item in manifest)
        if len(images) != len(labels) or len(images) != expected:
            errors.append(f"{split} image/label/manifest mismatch")
        instances = Counter()
        positives = Counter()
        for image_path in images:
            item = by_id.get(image_path.stem)
            if not item:
                errors.append(f"missing manifest record: {image_path.name}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"corrupt image {image_path.name}: {exc}")
                continue
            if sha256_file(image_path) != item["sha256"]:
                errors.append(f"SHA mismatch: {image_path.name}")
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            sample_classes = set()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                try:
                    parsed = parse_yolo_line(line, allowed_classes=set(EXPECTED_NAMES))
                except ValueError as exc:
                    errors.append(f"invalid label {label_path}: {exc}")
                    continue
                instances[parsed.class_id] += 1
                sample_classes.add(parsed.class_id)
            positives.update(sample_classes)
        if any(instances[class_id] == 0 for class_id in EXPECTED_NAMES):
            errors.append(f"{split} misses a class")
        split_report[split] = {
            "images": len(images),
            "instances": {EXPECTED_NAMES[key]: instances[key] for key in EXPECTED_NAMES},
            "positive_images": {EXPECTED_NAMES[key]: positives[key] for key in EXPECTED_NAMES},
        }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "task": "OCCUPANT_UPPER_BODY_ROI_DETECTION",
        "manifest_samples": len(manifest),
        "manifest_sha256": stable_json_hash(manifest),
        "splits": split_report,
        "overlaps": overlap,
        "errors": errors,
        "policy": "Integrity/isolation pass is not human semantic approval.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V2 specialist seatbelt data")
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("datasets/derived/seatbelt_v2_balanced")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/seatbelt_v2/audit.json"))
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
