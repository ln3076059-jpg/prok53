from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from training.common import YoloLabel, sha256_file, stable_json_hash
from training.pretrain_review_v2 import MIN_UNCERTAIN_COUNTS
from training.uncertain_visual_decisions import load_selected_uncertain_rows

SPLITS = ("train", "val", "test")
REVIEW_USAGE = "MODEL_ASSISTED_PENDING_APPROVAL"
CLASSIFIER_NAMES = {
    0: "seatbelt_fastened",
    1: "seatbelt_unfastened",
    2: "uncertain_or_occluded",
}


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _reviewed(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"model-assisted review did not pass: {path}")
    return [item for item in payload["samples"] if item["proposal_review"]["training_eligible"]]


def _source_map(dataset: Path) -> dict[str, dict]:
    return {item["sample_id"]: item for item in _jsonl(dataset / "manifest.jsonl")}


def _balanced_splits(rows: list[dict], source: dict[str, dict]) -> dict[str, str]:
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    groups: dict[str, list[dict]] = {}
    for item in rows:
        source_item = source.get(item["sample_id"])
        if source_item is None:
            raise ValueError(f"reviewed sample is absent from source manifest: {item['sample_id']}")
        group = str(source_item.get("effective_group_id") or source_item["source_group_id"])
        groups.setdefault(group, []).append(item)
    image_counts = Counter()
    box_counts = Counter()
    assignments = {}

    # Distribute groups containing positives independently from pure-negative groups.
    # Otherwise the positive-heavy groups arrive first and can consume nearly the
    # entire train box budget before negative groups balance only image counts.
    strata = {"positive": [], "negative": []}
    for group, items in groups.items():
        component_boxes = sum(len(item.get("annotations", [])) for item in items)
        key = "positive" if component_boxes else "negative"
        strata[key].append((group, items, component_boxes))

    for stratum in ("positive", "negative"):
        components = strata[stratum]
        stratum_images = sum(len(items) for _, items, _ in components)
        stratum_boxes = sum(boxes for _, _, boxes in components)
        local_images = Counter()
        local_boxes = Counter()
        ordered = sorted(
            components,
            key=lambda value: (len(value[1]), value[2], value[0]),
            reverse=True,
        )
        for _, items, component_boxes in ordered:
            fill_ratios = {}
            for candidate in SPLITS:
                image_target = max(targets[candidate] * stratum_images, 1)
                image_ratio = (local_images[candidate] + len(items)) / image_target
                if stratum_boxes:
                    box_target = max(targets[candidate] * stratum_boxes, 1)
                    box_ratio = (local_boxes[candidate] + component_boxes) / box_target
                else:
                    box_ratio = image_ratio
                # Minimize the worst normalized fill, then the total fill. The
                # split index is only a deterministic tie breaker.
                fill_ratios[candidate] = (
                    max(image_ratio, box_ratio),
                    image_ratio + box_ratio,
                    SPLITS.index(candidate),
                )

            split = min(SPLITS, key=fill_ratios.__getitem__)
            for item in items:
                assignments[item["sample_id"]] = split
            local_images[split] += len(items)
            local_boxes[split] += component_boxes
            image_counts[split] += len(items)
            box_counts[split] += component_boxes
    return assignments


def _detection_dataset(
    rows: list[dict],
    output: Path,
    splits: dict[str, str],
    class_names: dict[int, str],
    transform_to_roi: bool,
    source: dict[str, dict],
) -> dict:
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "labels" / split).mkdir(parents=True, exist_ok=False)
    manifest = []
    images = Counter()
    boxes = Counter()
    for item in rows:
        sample_id = item["sample_id"]
        split = splits.get(sample_id)
        if split not in SPLITS:
            raise ValueError(f"reviewed sample lacks inherited split: {sample_id}")
        source_image = Path(item["image_path"])
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        target_image = output / "images" / split / f"{sample_id}{source_image.suffix.lower()}"
        target_label = output / "labels" / split / f"{sample_id}.txt"
        shutil.copy2(source_image, target_image)
        labels = []
        for annotation in item.get("annotations", []):
            source_class = int(annotation["class_id"])
            target_class = 0 if transform_to_roi else source_class
            label = YoloLabel(target_class, *(float(value) for value in annotation["yolo"]))
            label.validate()
            labels.append(
                f"{label.class_id} {label.x_center!r} {label.y_center!r} "
                f"{label.width!r} {label.height!r}"
            )
        target_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        images[split] += 1
        boxes[split] += len(labels)
        review = item["proposal_review"]
        source_item = source[sample_id]
        manifest.append(
            {
                "sample_id": sample_id,
                "source_id": item.get("source_id"),
                "source_asset_id": item.get("source_asset_id"),
                "source_group_id": item.get("source_group_id"),
                "effective_group_id": source_item.get("effective_group_id")
                or source_item["source_group_id"],
                "split": split,
                "usage": REVIEW_USAGE,
                "reviewer_type": "AUTOMATED",
                "reviewer_model": review["reviewer_model"],
                "proposal_review_tier": review["tier"],
                "proposal_review_confidence": review["confidence"],
                "proposal_review_sha256": stable_json_hash(review),
                "human_review_status": "NOT_HUMAN_VERIFIED",
                "governance_eligible": False,
                "box_instances": len(labels),
                "sha256": sha256_file(target_image),
                "video_id_suggestion": review.get("metadata_suggestions", {}).get("video_id"),
                "condition_suggestions": review.get("metadata_suggestions", {}).get(
                    "conditions", []
                ),
            }
        )
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n",
        encoding="utf-8",
    )
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                # The training runner resolves this against the unpacked dataset.
                # Keeping it relative prevents host-specific paths in portable bundles.
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": class_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "images": {split: images[split] for split in SPLITS},
        "boxes": {split: boxes[split] for split in SPLITS},
        "manifest_sha256": stable_json_hash(manifest),
    }


def _crop(image: Image.Image, yolo: list[float], padding: float) -> Image.Image:
    width, height = image.size
    x, y, box_width, box_height = (float(value) for value in yolo)
    half_width = box_width * (1 + 2 * padding) / 2
    half_height = box_height * (1 + 2 * padding) / 2
    box = (
        max(0, int((x - half_width) * width)),
        max(0, int((y - half_height) * height)),
        min(width, int((x + half_width) * width)),
        min(height, int((y + half_height) * height)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid crop box: {box}")
    return image.crop(box).convert("RGB")


def _classifier_dataset(
    seatbelt_rows: list[dict],
    uncertain_rows: list[dict],
    output: Path,
    seatbelt_splits: dict[str, str],
    seatbelt_source: dict[str, dict],
    padding: float = 0.08,
) -> dict:
    for split in SPLITS:
        for class_name in CLASSIFIER_NAMES.values():
            (output / split / class_name).mkdir(parents=True, exist_ok=False)
    manifest = []
    counts = {split: Counter() for split in SPLITS}
    uncertain_keys = {
        (str(item["source_sample_id"]), int(item.get("box_index", 0))) for item in uncertain_rows
    }

    def add(item: dict, split: str, annotation: dict, class_id: int, box_index: int) -> None:
        sample_id = item["sample_id"]
        source_key = str(item.get("source_sample_id", sample_id))
        safe_sample_id = source_key.replace(":", "_").replace("/", "_").replace("\\", "_")
        crop_id = f"{safe_sample_id}_box_{box_index}"
        source_image = Path(item["image_path"])
        with Image.open(source_image) as image:
            crop = _crop(image, annotation["yolo"], padding)
        target = output / split / CLASSIFIER_NAMES[class_id] / f"{crop_id}.jpg"
        crop.save(target, format="JPEG", quality=95)
        review = item["proposal_review"]
        source_item = seatbelt_source[source_key]
        manifest.append(
            {
                "sample_id": crop_id,
                "source_sample_id": source_key,
                "source_box_index": box_index,
                "source_group_id": item.get("source_group_id"),
                "effective_group_id": source_item.get("effective_group_id")
                or source_item["source_group_id"],
                "split": split,
                "class_id": class_id,
                "class_name": CLASSIFIER_NAMES[class_id],
                "image_path": str(target.relative_to(output)).replace("\\", "/"),
                "sha256": sha256_file(target),
                "yolo": annotation["yolo"],
                "usage": REVIEW_USAGE,
                "reviewer_type": "AUTOMATED",
                "reviewer_model": review["reviewer_model"],
                "proposal_review_tier": review["tier"],
                "proposal_review_confidence": review["confidence"],
                "human_review_status": "NOT_HUMAN_VERIFIED",
                "governance_eligible": False,
            }
        )
        counts[split][CLASSIFIER_NAMES[class_id]] += 1

    for item in seatbelt_rows:
        split = seatbelt_splits.get(item["sample_id"])
        if split not in SPLITS:
            raise ValueError(f"seatbelt sample lacks inherited split: {item['sample_id']}")
        for index, annotation in enumerate(item.get("annotations", [])):
            if (item["sample_id"], index) in uncertain_keys:
                continue
            source_class = int(annotation["class_id"])
            if source_class not in {1, 2}:
                continue
            add(item, split, annotation, source_class - 1, index)
    for item in uncertain_rows:
        split = item.get("split")
        if split not in SPLITS:
            raise ValueError(f"uncertain sample lacks split: {item['sample_id']}")
        annotations = item.get("annotations", [])
        if len(annotations) != 1:
            raise ValueError(f"uncertain candidate needs exactly one ROI: {item['sample_id']}")
        add(item, split, annotations[0], 2, 0)

    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n",
        encoding="utf-8",
    )
    state_classes_present = all(
        counts[split][name] > 0
        for split in SPLITS
        for name in ("seatbelt_fastened", "seatbelt_unfastened")
    )
    uncertainty_minimum_met = all(
        counts[split]["uncertain_or_occluded"] >= MIN_UNCERTAIN_COUNTS[split] for split in SPLITS
    )
    complete = state_classes_present and uncertainty_minimum_met
    report = {
        "status": "PRETRAIN_PENDING_APPROVAL_READY" if complete else "INCOMPLETE",
        "ready_for_training": complete,
        "ready_for_exploratory_training": complete,
        "ready_for_governed_training": False,
        "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "classes": CLASSIFIER_NAMES,
        "minimum_uncertain_counts": MIN_UNCERTAIN_COUNTS,
        "uncertainty_minimum_met": uncertainty_minimum_met,
        "uncertain_replacements": len(uncertain_keys),
        "padding": padding,
        "manifest_sha256": stable_json_hash(manifest),
        "policy": (
            "Model-assisted pending-approval crops are not human-approved governed ground truth."
        ),
    }
    (output / "CLASSIFIER_DATASET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build(root: Path) -> dict:
    if root.exists():
        raise FileExistsError(
            f"model-assisted pending-approval dataset root already exists: {root}"
        )
    phone_rows = [
        *_reviewed(Path("datasets/manifests/pretrain_pending_approval/v2_phone_positive.json")),
        *_reviewed(Path("datasets/manifests/pretrain_pending_approval/v2_phone_negative.json")),
    ]
    seatbelt_rows = _reviewed(
        Path("datasets/manifests/pretrain_pending_approval/v2_seatbelt_roi_state.json")
    )
    uncertain_decisions = Path(
        "datasets/manifests/pretrain_pending_approval/"
        "v2_seatbelt_uncertain_visual_decisions.json"
    )
    uncertain_rows = load_selected_uncertain_rows(
        Path("datasets/manifests/pretrain_pending_approval/v2_seatbelt_uncertain.json"),
        uncertain_decisions,
    )
    phone_source = _source_map(Path("datasets/derived/phone_bootstrap_v2"))
    phone = _detection_dataset(
        phone_rows,
        root / "phone_detector",
        _balanced_splits(phone_rows, phone_source),
        {0: "phone"},
        False,
        phone_source,
    )
    seatbelt_source = _source_map(Path("datasets/derived/seatbelt_v2_balanced"))
    seatbelt_splits = {sample_id: item["split"] for sample_id, item in seatbelt_source.items()}
    seatbelt = _detection_dataset(
        seatbelt_rows,
        root / "seatbelt_detector",
        seatbelt_splits,
        {0: "occupant_upper_body"},
        True,
        seatbelt_source,
    )
    classifier = _classifier_dataset(
        seatbelt_rows,
        uncertain_rows,
        root / "seatbelt_classifier",
        seatbelt_splits,
        seatbelt_source,
    )
    report = {
        "schema_version": 1,
        "status": (
            "PRETRAIN_PENDING_APPROVAL_READY"
            if classifier["ready_for_exploratory_training"]
            else "INCOMPLETE"
        ),
        "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
        "training_ready": classifier["ready_for_exploratory_training"],
        "governed_training_ready": False,
        "uncertainty_visual_decisions": {
            "path": str(uncertain_decisions),
            "sha256": sha256_file(uncertain_decisions),
            "selected": len(uncertain_rows),
        },
        "datasets": {
            "phone_detector": phone,
            "seatbelt_detector": seatbelt,
            "seatbelt_classifier": classifier,
        },
        "policy": (
            "The detector proposal lanes may train in isolation. The complete suite remains "
            "blocked until the classifier uncertainty-evidence minimum is satisfied. "
            "Production/governed claims require human verification."
            if not classifier["ready_for_exploratory_training"]
            else (
                "May train the exploratory suite; production/governed claims require "
                "human verification."
            )
        ),
    }
    (root / "PRETRAIN_PENDING_APPROVAL_DATASET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-assisted pending-approval exploratory V2 datasets"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/derived/v2_pretrain_pending_approval")
    )
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["training_ready"] else 1)


if __name__ == "__main__":
    main()
