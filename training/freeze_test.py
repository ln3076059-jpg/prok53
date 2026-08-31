from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from training.common import CLASS_NAMES, read_labels, sha256_file, stable_json_hash
from training.split_dataset import validate_isolation


def freeze(root: Path, output: Path) -> dict:
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line]
    isolation = validate_isolation(manifest)
    if isolation["status"] != "PASS":
        raise ValueError(f"cannot freeze leaking split: {isolation}")
    test_items = [item for item in manifest if item["split"] == "test"]
    if not test_items:
        raise ValueError("test split is empty")
    counts = Counter()
    file_hashes = {}
    for item in test_items:
        label_path = root / "labels" / "test" / f"{item['sample_id']}.txt"
        for label in read_labels(label_path):
            counts[CLASS_NAMES[label.class_id]] += 1
        image_candidates = list((root / "images" / "test").glob(f"{item['sample_id']}.*"))
        if len(image_candidates) != 1:
            raise ValueError(f"expected one image for {item['sample_id']}")
        file_hashes[item["sample_id"]] = sha256_file(image_candidates[0])
    frozen = {
        "status": "FROZEN",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": stable_json_hash(file_hashes),
        "manifest_sha256": stable_json_hash(manifest),
        "test_groups": sorted({item["effective_group_id"] for item in test_items}),
        "test_samples": len(test_items),
        "class_counts": dict(counts),
        "subject_isolation": isolation["subject_isolation"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/governed/mc3_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/manifests/test_frozen.json"))
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()

