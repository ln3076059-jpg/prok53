from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from training.common import CLASS_NAMES, read_labels

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else round(float(np.percentile(values, q)), 3)


def analyze(root: Path) -> dict:
    report = {
        "status": "PASS",
        "images": 0,
        "brightness": [],
        "contrast": [],
        "laplacian_variance": [],
        "object_short_side_pixels": defaultdict(list),
        "risk_counts": Counter(),
        "errors": [],
    }
    for split in ("train", "val", "test"):
        for image_path in sorted((root / "images" / split).glob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                report["errors"].append(f"cannot decode {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            report["brightness"].append(brightness)
            report["contrast"].append(contrast)
            report["laplacian_variance"].append(blur_score)
            report["images"] += 1
            if brightness < 45:
                report["risk_counts"]["very_dark"] += 1
            if brightness > 215:
                report["risk_counts"]["overexposed"] += 1
            if contrast < 25:
                report["risk_counts"]["low_contrast"] += 1
            if blur_score < 50:
                report["risk_counts"]["likely_blur"] += 1
            height, width = gray.shape
            for label in read_labels(root / "labels" / split / f"{image_path.stem}.txt"):
                short_side = min(label.width * width, label.height * height)
                report["object_short_side_pixels"][CLASS_NAMES[label.class_id]].append(short_side)
                if short_side < 8:
                    report["risk_counts"][f"tiny_under_8px_{CLASS_NAMES[label.class_id]}"] += 1
    if report["errors"]:
        report["status"] = "FAIL"
    summary = {
        "status": report["status"],
        "images": report["images"],
        "image_statistics": {
            "brightness_p05_p50_p95": [percentile(report["brightness"], q) for q in (5, 50, 95)],
            "contrast_p05_p50_p95": [percentile(report["contrast"], q) for q in (5, 50, 95)],
            "laplacian_variance_p05_p50_p95": [percentile(report["laplacian_variance"], q) for q in (5, 50, 95)],
        },
        "object_short_side_pixels": {
            name: {
                "p05": percentile(values, 5),
                "p50": percentile(values, 50),
                "count": len(values),
            }
            for name, values in report["object_short_side_pixels"].items()
        },
        "risk_counts": dict(report["risk_counts"]),
        "errors": report["errors"],
        "interpretation": "Thresholds flag review candidates; they do not automatically delete data.",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure small-object and camera-condition risks")
    parser.add_argument("root", type=Path, nargs="?", default=Path("datasets/derived/mc3_v1_aug"))
    parser.add_argument("--output", type=Path, default=Path("reports/cv_quality.json"))
    args = parser.parse_args()
    report = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
