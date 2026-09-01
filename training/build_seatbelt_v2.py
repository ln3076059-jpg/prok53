from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from training.common import stable_json_hash

SOURCE_TO_STATE = {1: "seatbelt_fastened", 2: "seatbelt_unfastened"}
SPECIALIST_NAMES = {0: "occupant_upper_body"}
SPLITS = ("train", "val", "test")


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _find_image(root: Path, split: str, sample_id: str) -> Path:
    matches = [path for path in (root / "images" / split).glob(f"{sample_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one image for {split}/{sample_id}, found {len(matches)}")
    return matches[0]


def _specialist_labels(path: Path) -> list[str]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        source_class = int(parts[0])
        if source_class in SOURCE_TO_STATE:
            output.append(" ".join(["0", *parts[1:]]))
    return output


def build(
    source: Path,
    output: Path,
) -> dict:
    """Build a label-complete upper-body ROI detector dataset.

    Only records from the seatbelt-state source are admitted. This prevents phone-only
    images with missing belt-state annotations from teaching the seatbelt detector that
    visible occupants are background. No image is duplicated or synthetically oversampled.
    """
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest = _read_manifest(source / "manifest.jsonl")
    selected = [item for item in manifest if item["source_id"] == "roboflow_seatbelttraining_v4"]
    if not selected:
        raise ValueError("source contains no seatbelt-state records")

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "labels" / split).mkdir(parents=True, exist_ok=False)

    records: list[dict] = []
    for item in selected:
        split = item["split"]
        source_image = _find_image(source, split, item["sample_id"])
        source_label = source / "labels" / split / f"{item['sample_id']}.txt"
        labels = _specialist_labels(source_label)
        if not labels:
            raise ValueError(f"seatbelt source record has no state label: {item['sample_id']}")
        target_image = output / "images" / split / source_image.name
        target_label = output / "labels" / split / f"{item['sample_id']}.txt"
        shutil.copy2(source_image, target_image)
        target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
        class_counts = Counter(int(line.split()[0]) for line in labels)
        source_states = [
            SOURCE_TO_STATE[int(line.split()[0])]
            for line in source_label.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(line.split()[0]) in SOURCE_TO_STATE
        ]
        if len(source_states) != len(labels):
            raise ValueError(f"state/ROI mismatch: {item['sample_id']}")
        records.append(
            {
                **item,
                "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
                "specialist_class_schema": SPECIALIST_NAMES,
                "label_path": str(target_label.resolve()),
                "image_path": str(target_image.resolve()),
                "seatbelt_states": source_states,
            }
        )

    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n",
        encoding="utf-8",
    )
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(SPECIALIST_NAMES),
        "names": SPECIALIST_NAMES,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    image_counts = Counter(item["split"] for item in records)
    instances = {split: Counter() for split in SPLITS}
    positives = {split: Counter() for split in SPLITS}
    for item in records:
        counts = {int(key): value for key, value in item["class_counts"].items()}
        instances[item["split"]].update(counts)
        positives[item["split"]].update(counts.keys())
    report = {
        "status": "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA",
        "task": "OCCUPANT_UPPER_BODY_ROI_DETECTION",
        "source_dataset": str(source.resolve()),
        "source_manifest_sha256": stable_json_hash(manifest),
        "manifest_sha256": stable_json_hash(records),
        "classes": SPECIALIST_NAMES,
        "images": {split: image_counts[split] for split in SPLITS},
        "instances": {
            split: {SPECIALIST_NAMES[key]: instances[split][key] for key in SPECIALIST_NAMES}
            for split in SPLITS
        },
        "positive_images": {
            split: {SPECIALIST_NAMES[key]: positives[split][key] for key in SPECIALIST_NAMES}
            for split in SPLITS
        },
        "seatbelt_state_instances": dict(
            Counter(state for item in records for state in item["seatbelt_states"])
        ),
        "augmentation": "NONE_NO_DUPLICATE_OVERSAMPLING",
        "required_diversity": ["independent_vehicle", "independent_person", "independent_camera", "lighting", "viewpoint"],
        "missing_label_control": "PHONE_ONLY_AND_OTHER_PARTIAL_LABEL_SOURCES_EXCLUDED",
        "human_review_status": "PENDING",
    }
    (output / "V2_DATASET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V2 seatbelt specialist dataset")
    parser.add_argument(
        "--source", type=Path, default=Path("datasets/derived/mc_bootstrap_v2_6500")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/derived/seatbelt_v2_balanced")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
