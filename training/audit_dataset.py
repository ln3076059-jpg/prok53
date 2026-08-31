from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from training.common import CLASS_NAMES, hamming_distance, read_labels, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def audit_dataset(root: Path, near_threshold: int = 6) -> dict:
    report: dict[str, object] = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "splits": {},
        "class_instances": Counter(),
        "exact_duplicate_groups": [],
        "near_duplicate_pairs": [],
    }
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    phashes: list[tuple[str, str, str]] = []
    manifest_path = root / "manifest.jsonl"
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            item = json.loads(line)
            manifest[item["sample_id"]] = item
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        split_counts = Counter()
        if not image_dir.exists():
            report["warnings"].append(f"missing split directory: {image_dir}")
            report["splits"][split] = {"images": 0, "labels": 0}
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                report["errors"].append(f"corrupt image {image_path}: {exc}")
                continue
            hashes[sha256_file(image_path)].append(f"{split}/{image_path.name}")
            item = manifest.get(image_path.stem, {})
            if item.get("phash"):
                phashes.append((split, image_path.name, item["phash"]))
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                report["warnings"].append(f"missing label: {label_path}")
                continue
            try:
                labels = read_labels(label_path)
                serialized = [tuple(label.__dict__.values()) for label in labels]
                if len(serialized) != len(set(serialized)):
                    report["errors"].append(f"duplicate label lines: {label_path}")
                for label in labels:
                    split_counts[CLASS_NAMES[label.class_id]] += 1
                    report["class_instances"][CLASS_NAMES[label.class_id]] += 1
            except ValueError as exc:
                report["errors"].append(f"invalid label {label_path}: {exc}")
        report["splits"][split] = {"images": len(images), "class_instances": dict(split_counts)}
    report["exact_duplicate_groups"] = [paths for paths in hashes.values() if len(paths) > 1]
    for index, (left_split, left_name, left_hash) in enumerate(phashes):
        for right_split, right_name, right_hash in phashes[index + 1 :]:
            if left_split != right_split and hamming_distance(left_hash, right_hash) <= near_threshold:
                report["near_duplicate_pairs"].append([f"{left_split}/{left_name}", f"{right_split}/{right_name}"])
    if report["exact_duplicate_groups"]:
        report["errors"].append("exact duplicates exist")
    if report["near_duplicate_pairs"]:
        report["errors"].append("near duplicates cross split boundaries")
    if report["errors"]:
        report["status"] = "FAIL"
    report["class_instances"] = dict(report["class_instances"])
    return report


def render_markdown(report: dict) -> str:
    lines = ["# Data Quality Report", "", f"Status: **{report['status']}**", "", "## Split summary", ""]
    for split, values in report["splits"].items():
        lines.append(f"- `{split}`: {values}")
    lines.extend(["", "## Class instances", ""])
    for name, count in report["class_instances"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Errors", ""] + [f"- {item}" for item in report["errors"]] or ["- None"])
    lines.extend(["", "## Warnings", ""] + [f"- {item}" for item in report["warnings"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("datasets/governed/mc3_v1"))
    parser.add_argument("--near-threshold", type=int, default=6)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = audit_dataset(args.root, args.near_threshold)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "data_quality.json").write_text(json.dumps(report, indent=2))
    (args.report_dir / "data_quality.md").write_text(render_markdown(report))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
