from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from training.common import perceptual_hash, read_labels, sha256_file, stable_json_hash

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def transform(image: np.ndarray, name: str, rng: random.Random) -> np.ndarray:
    if name == "low_light":
        gamma = rng.uniform(1.5, 2.2)
        table = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    if name == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    if name == "motion_blur":
        size = rng.choice([3, 5, 7])
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        return cv2.filter2D(image, -1, kernel)
    if name == "sensor_noise":
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(4, 10), image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "compression":
        height, width = image.shape[:2]
        factor = rng.uniform(0.45, 0.7)
        small = cv2.resize(image, (max(1, int(width * factor)), max(1, int(height * factor))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown transform: {name}")


def augment(source: Path, output: Path, seed: int, max_variants: int) -> dict:
    if output.exists():
        raise FileExistsError(f"derived dataset already exists: {output}")
    shutil.copytree(source, output)
    manifest_path = output / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    train_records = [item for item in manifest if item["split"] == "train"]
    class_totals = Counter()
    labels_by_sample = {}
    for record in train_records:
        labels = read_labels(output / "labels" / "train" / f"{record['sample_id']}.txt")
        labels_by_sample[record["sample_id"]] = labels
        class_totals.update(label.class_id for label in labels)
    if not class_totals:
        raise ValueError("training split has no labels")
    largest = max(class_totals.values())
    transforms = ["low_light", "clahe", "motion_blur", "sensor_noise", "compression"]
    derived = []
    for record in train_records:
        labels = labels_by_sample[record["sample_id"]]
        if not labels:
            continue
        rarest = min(class_totals[label.class_id] for label in labels)
        variants = min(max_variants, max(0, math.ceil(largest / max(rarest, 1)) - 1))
        rng = random.Random(f"{seed}:{record['sample_id']}")
        image_candidates = list((output / "images" / "train").glob(f"{record['sample_id']}.*"))
        if len(image_candidates) != 1:
            raise ValueError(f"expected one train image for {record['sample_id']}")
        image = cv2.imread(str(image_candidates[0]))
        if image is None:
            raise ValueError(f"cannot decode {image_candidates[0]}")
        label_source = output / "labels" / "train" / f"{record['sample_id']}.txt"
        for index in range(variants):
            transform_name = transforms[(rng.randrange(len(transforms)) + index) % len(transforms)]
            sample_id = f"{record['sample_id']}_aug_{index}_{transform_name}"
            image_target = output / "images" / "train" / f"{sample_id}.jpg"
            label_target = output / "labels" / "train" / f"{sample_id}.txt"
            if not cv2.imwrite(str(image_target), transform(image, transform_name, rng), [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f"failed to write {image_target}")
            shutil.copy2(label_source, label_target)
            derived.append(
                {
                    **record,
                    "sample_id": sample_id,
                    "derived_from_sample_id": record["sample_id"],
                    "augmentation": transform_name,
                    "sha256": sha256_file(image_target),
                    "phash": perceptual_hash(image_target),
                    "reviewed_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "annotation_provenance": "human_reviewed_box_photometric_derivative",
                }
            )
    combined = manifest + derived
    manifest_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in combined) + "\n", encoding="utf-8")
    metadata_path = output / "augmentation_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "train_only": True,
                "bbox_geometry_changed": False,
                "source_dataset": str(source.resolve()),
                "source_manifest_sha256": stable_json_hash(manifest),
                "derived_samples": len(derived),
                "transforms": transforms,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"output": str(output), "original_samples": len(manifest), "derived_train_samples": len(derived)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic train-only camera-condition augmentation")
    parser.add_argument("--source", type=Path, default=Path("datasets/governed/mc3_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/derived/mc3_v1_aug"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-variants", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(augment(args.source, args.output, args.seed, args.max_variants), indent=2))


if __name__ == "__main__":
    main()
