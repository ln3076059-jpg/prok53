from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageStat


def _manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _image(root: Path, item: dict) -> Path:
    matches = list((root / "images" / item["split"]).glob(f"{item['sample_id']}.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one image for {item['sample_id']}")
    return matches[0]


def _quality(path: Path) -> tuple[float, dict]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((96, 96))
        stats = ImageStat.Stat(gray)
        brightness = float(stats.mean[0])
        contrast = float(stats.stddev[0])
    decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError(f"cannot decode {path}")
    sharpness = float(cv2.Laplacian(decoded, cv2.CV_64F).var())
    low_light = max(0.0, (85.0 - brightness) / 85.0)
    low_contrast = max(0.0, (40.0 - contrast) / 40.0)
    blur = max(0.0, (140.0 - sharpness) / 140.0)
    return low_light + low_contrast + blur, {
        "brightness": brightness,
        "contrast": contrast,
        "laplacian_variance": sharpness,
    }


def build(root: Path, output: Path, targets: dict[str, int]) -> dict:
    records = _manifest(root / "manifest.jsonl")
    candidates: defaultdict[str, list[dict]] = defaultdict(list)
    for item in records:
        image_path = _image(root, item)
        quality_score, quality = _quality(image_path)
        label_path = root / "labels" / item["split"] / f"{item['sample_id']}.txt"
        for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            yolo = [float(value) for value in parts[1:]]
            x, y, width, height = yolo
            border_margin = min(x - width / 2, y - height / 2, 1 - x - width / 2, 1 - y - height / 2)
            border_score = max(0.0, (0.04 - border_margin) / 0.04)
            score = quality_score + border_score
            candidates[item["split"]].append(
                {
                    "candidate_id": f"{item['sample_id']}:box:{box_index}",
                    "sample_id": item["sample_id"],
                    "box_index": box_index,
                    "split": item["split"],
                    "image_path": str(image_path.resolve()),
                    "sha256": item["sha256"],
                    "yolo": yolo,
                    "source_group_id": item["source_group_id"],
                    "effective_group_id": item["effective_group_id"],
                    "quality": quality,
                    "uncertainty_priority_score": score,
                }
            )
    selected = []
    for split, target in targets.items():
        best_by_group = {}
        for item in candidates[split]:
            group = item["source_group_id"]
            current = best_by_group.get(group)
            if current is None or item["uncertainty_priority_score"] > current["uncertainty_priority_score"]:
                best_by_group[group] = item
        ranked = sorted(
            best_by_group.values(),
            key=lambda item: (-item["uncertainty_priority_score"], item["candidate_id"]),
        )[:target]
        for item in ranked:
            selected.append(
                {
                    **item,
                    "review_status": "PENDING",
                    "allowed_decisions": [
                        "SEATBELT_FASTENED",
                        "SEATBELT_UNFASTENED",
                        "UNCERTAIN_OR_OCCLUDED",
                        "REJECT_BAD_ROI",
                    ],
                    "automatic_training_label": None,
                    "annotation_rule": "Missing/invisible belt evidence must be UNCERTAIN_OR_OCCLUDED, not UNFASTENED.",
                }
            )
    report = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "selection_policy": "ONE_DIFFICULT_BOX_PER_SOURCE_VIDEO_GROUP_MAX",
        "targets": targets,
        "selected_counts": {
            split: sum(item["split"] == split for item in selected) for split in targets
        },
        "selected": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize difficult belt crops for uncertainty review")
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/derived/seatbelt_v2_balanced")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/manifests/seatbelt_uncertain_review_v2.json")
    )
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()
    report = build(
        args.root, args.output, {"train": args.train, "val": args.val, "test": args.test}
    )
    print(json.dumps({key: value for key, value in report.items() if key != "selected"}, indent=2))


if __name__ == "__main__":
    main()
