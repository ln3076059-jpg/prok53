from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageFilter, ImageStat

from training.build_mc_bootstrap import CLASS_NAMES, SPLITS, STATUS
from training.common import parse_yolo_line, stable_json_hash

TARGET_TRAIN = 6500
TARGET_FASTENED = 2400
TARGET_NEGATIVE = 1039
DMS_CLASS_NAMES = {
    0: "open_eye",
    1: "closed_eye",
    2: "cigarette",
    3: "phone",
    4: "seatbelt",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def image_path(root: Path, item: dict) -> Path:
    candidates = list((root / "images" / item["split"]).glob(f"{item['sample_id']}.*"))
    if len(candidates) != 1:
        raise ValueError(f"expected one image for {item['sample_id']}, found {len(candidates)}")
    return candidates[0]


def source_context(item: dict, dms_root: Path) -> str:
    if item["source_id"] != "kaggle_habbas_dms_v1":
        return "seatbelt_state"
    source_image = dms_root / Path(item["source_asset_id"].replace("\\", "/"))
    source_label = source_image.parent.parent / "labels" / f"{source_image.stem}.txt"
    class_ids = sorted(
        {int(line.split()[0]) for line in source_label.read_text(encoding="utf-8").splitlines() if line}
    )
    return "+".join(DMS_CLASS_NAMES[class_id] for class_id in class_ids)


def visual_features(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        gray = image.convert("L").resize((64, 64))
        stats = ImageStat.Stat(gray)
        edge_variance = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0]
        return {
            "width": width,
            "height": height,
            "brightness": float(stats.mean[0]),
            "contrast": float(stats.stddev[0]),
            "sharpness": float(edge_variance),
        }


def label_features(label_path: Path) -> dict:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_yolo_line(line)
        boxes.append(
            {
                "class_id": parsed.class_id,
                "area": parsed.width * parsed.height,
                "margin": min(
                    parsed.x_center - parsed.width / 2,
                    parsed.y_center - parsed.height / 2,
                    1 - parsed.x_center - parsed.width / 2,
                    1 - parsed.y_center - parsed.height / 2,
                ),
            }
        )
    return {
        "boxes": len(boxes),
        "min_box_area": min((box["area"] for box in boxes), default=None),
        "min_box_margin": min((box["margin"] for box in boxes), default=None),
    }


def rank_map(values: list[float], reverse: bool = False) -> dict[float, float]:
    ordered = sorted(values, reverse=reverse)
    if len(ordered) <= 1:
        return {value: 1.0 for value in ordered}
    ranks = {}
    for position, value in enumerate(ordered):
        ranks.setdefault(value, position / (len(ordered) - 1))
    return ranks


def add_difficulty(records: list[dict]) -> None:
    brightness_extremes = [abs(item["brightness"] - 127.5) for item in records]
    contrasts = [item["contrast"] for item in records]
    sharpness = [item["sharpness"] for item in records]
    resolutions = [min(item["width"], item["height"]) for item in records]
    box_areas = [item["min_box_area"] for item in records if item["min_box_area"] is not None]
    brightness_ranks = rank_map(brightness_extremes)
    contrast_ranks = rank_map(contrasts, reverse=True)
    sharpness_ranks = rank_map(sharpness, reverse=True)
    resolution_ranks = rank_map(resolutions, reverse=True)
    box_area_ranks = rank_map(box_areas, reverse=True)
    for item in records:
        visual_hardness = (
            brightness_ranks[abs(item["brightness"] - 127.5)]
            + contrast_ranks[item["contrast"]]
            + sharpness_ranks[item["sharpness"]]
            + resolution_ranks[min(item["width"], item["height"])]
        ) / 4
        box_hardness = 0.0
        if item["min_box_area"] is not None:
            box_hardness = box_area_ranks[item["min_box_area"]]
            if item["min_box_margin"] is not None and item["min_box_margin"] < 0.02:
                box_hardness += 0.35
            if item["clipped_boxes"]:
                box_hardness += 0.35
            if item["boxes"] > 1:
                box_hardness += min(0.25, 0.05 * (item["boxes"] - 1))
        distractor_bonus = 0.0
        if "cigarette" in item["source_context"]:
            distractor_bonus += 0.35
        if "closed_eye" in item["source_context"]:
            distractor_bonus += 0.15
        item["difficulty_score"] = round(2.5 * box_hardness + visual_hardness + distractor_bonus, 8)
        item["visual_context"] = ":".join(
            [
                "dark" if item["brightness"] < 70 else "bright" if item["brightness"] > 185 else "normal_light",
                "low_contrast" if item["contrast"] < 32 else "normal_contrast",
                "blurred" if item["sharpness"] < 180 else "normal_sharpness",
                "portrait" if item["height"] > item["width"] * 1.15 else "landscape" if item["width"] > item["height"] * 1.15 else "square",
            ]
        )


def _best_by(records: list[dict], key: str) -> list[dict]:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for item in records:
        groups[str(item[key])].append(item)
    return [
        max(values, key=lambda item: (item["difficulty_score"], item["sample_id"]))
        for _, values in sorted(groups.items())
    ]


def select_with_coverage(records: list[dict], target: int, prefix: str) -> dict[str, list[str]]:
    if target > len(records):
        raise ValueError(f"target {target} exceeds {prefix} candidates {len(records)}")
    selected: dict[str, list[str]] = {}

    def add(items: list[dict], reason: str) -> None:
        for item in sorted(items, key=lambda value: (-value["difficulty_score"], value["sample_id"])):
            if len(selected) >= target:
                return
            selected.setdefault(item["sample_id"], []).append(reason)

    add(_best_by(records, "source_context"), f"{prefix}_SOURCE_CONTEXT")
    add(_best_by(records, "source_group_id"), f"{prefix}_SOURCE_GROUP_REPRESENTATIVE")
    add(_best_by(records, "effective_group_id"), f"{prefix}_NEAR_COMPONENT_REPRESENTATIVE")
    add(_best_by(records, "visual_context"), f"{prefix}_VISUAL_CONTEXT")
    add(records, f"{prefix}_HARDNESS_RANK")
    if len(selected) != target:
        raise AssertionError(f"selected {len(selected)} {prefix} images, expected {target}")
    return selected


def copy_or_link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_subset(source: Path, output: Path, dms_root: Path, target_train: int = TARGET_TRAIN) -> dict:
    if output.exists():
        raise FileExistsError(f"subset output already exists: {output}")
    manifest = read_jsonl(source / "manifest.jsonl")
    train_items = [dict(item) for item in manifest if item["split"] == "train"]
    for item in train_items:
        path = image_path(source, item)
        item.update(visual_features(path))
        item.update(label_features(source / "labels" / "train" / f"{item['sample_id']}.txt"))
        item["source_context"] = source_context(item, dms_root)
    add_difficulty(train_items)

    phone = [item for item in train_items if "0" in item["class_counts"]]
    unfastened = [item for item in train_items if "2" in item["class_counts"]]
    fastened = [item for item in train_items if "1" in item["class_counts"]]
    negatives = [item for item in train_items if not item["class_counts"]]
    if len(phone) != 1857 or len(unfastened) != 1216:
        raise ValueError("source train split no longer matches the audited minority-class counts")

    reasons: dict[str, list[str]] = {
        item["sample_id"]: ["ALL_PHONE_POSITIVES"] for item in phone
    }
    for item in unfastened:
        reasons.setdefault(item["sample_id"], []).append("ALL_UNFASTENED_POSITIVES")
    fastened_already = sum(item["sample_id"] in reasons for item in fastened)
    fastened_candidates = [item for item in fastened if item["sample_id"] not in reasons]
    reasons.update(
        select_with_coverage(
            fastened_candidates,
            TARGET_FASTENED - fastened_already,
            "FASTENED",
        )
    )
    reasons.update(select_with_coverage(negatives, TARGET_NEGATIVE, "HARD_NEGATIVE"))
    selected_ids = set(reasons)
    if len(selected_ids) != target_train:
        raise AssertionError(f"selected {len(selected_ids)} train images, expected {target_train}")

    record_by_id = {item["sample_id"]: item for item in train_items}
    output.mkdir(parents=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    subset_manifest = []
    for original in manifest:
        if original["split"] == "train" and original["sample_id"] not in selected_ids:
            continue
        item = dict(original)
        if item["split"] == "train":
            enriched = record_by_id[item["sample_id"]]
            item["subset_selection"] = {
                "reasons": reasons[item["sample_id"]],
                "difficulty_score": enriched["difficulty_score"],
                "source_context": enriched["source_context"],
                "visual_context": enriched["visual_context"],
            }
        else:
            item["subset_selection"] = {"reasons": ["LOCKED_SPLIT_PRESERVED"]}
        source_image = image_path(source, original)
        target_image = output / "images" / item["split"] / source_image.name
        source_label = source / "labels" / item["split"] / f"{item['sample_id']}.txt"
        target_label = output / "labels" / item["split"] / source_label.name
        copy_or_link(source_image, target_image)
        copy_or_link(source_label, target_label)
        subset_manifest.append(item)

    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in subset_manifest) + "\n",
        encoding="utf-8",
    )
    data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(output.resolve())
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    split_images = Counter(item["split"] for item in subset_manifest)
    class_instances = {split: Counter() for split in SPLITS}
    positive_images = {split: Counter() for split in SPLITS}
    source_images = {split: Counter() for split in SPLITS}
    components = {split: set() for split in SPLITS}
    for item in subset_manifest:
        counts = {int(key): value for key, value in item["class_counts"].items()}
        class_instances[item["split"]].update(counts)
        positive_images[item["split"]].update(counts.keys())
        source_images[item["split"]][item["source_id"]] += 1
        components[item["split"]].add(item["effective_group_id"])
    summary = {
        "status": STATUS,
        "classes": CLASS_NAMES,
        "images": {split: split_images[split] for split in SPLITS},
        "class_instances": {
            split: {CLASS_NAMES[key]: class_instances[split][key] for key in CLASS_NAMES}
            for split in SPLITS
        },
        "positive_images": {
            split: {CLASS_NAMES[key]: positive_images[split][key] for key in CLASS_NAMES}
            for split in SPLITS
        },
        "source_images": {split: dict(source_images[split]) for split in SPLITS},
        "component_overlap": {
            "train_val": len(components["train"] & components["val"]),
            "train_test": len(components["train"] & components["test"]),
            "val_test": len(components["val"] & components["test"]),
        },
        "selection": {
            "policy": "DIFFICULTY_AND_CONTEXT_PRESERVING_TRAIN_SUBSET",
            "target_train_images": target_train,
            "all_phone_positive_images_retained": len(phone),
            "all_unfastened_positive_images_retained": len(unfastened),
            "target_fastened_positive_images": TARGET_FASTENED,
            "target_hard_negative_images": TARGET_NEGATIVE,
            "parent_manifest_sha256": stable_json_hash(manifest),
            "locked_validation_and_test_preserved": True,
        },
        "policy": "Bootstrap/proposal training only. Human review remains mandatory before final MC_001 training or accuracy claims.",
    }
    (output / "BOOTSTRAP_MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "SELECTION_REPORT.json").write_text(
        json.dumps(
            {
                "summary": summary["selection"],
                "selected_train_sha256": stable_json_hash(
                    sorted(item["sample_id"] for item in subset_manifest if item["split"] == "train")
                ),
                "source_group_coverage": {
                    "before": len({item["source_group_id"] for item in train_items}),
                    "after": len(
                        {
                            item["source_group_id"]
                            for item in train_items
                            if item["sample_id"] in selected_ids
                        }
                    ),
                },
                "source_context_counts": Counter(
                    record_by_id[sample_id]["source_context"] for sample_id in selected_ids
                ),
                "visual_context_counts": Counter(
                    record_by_id[sample_id]["visual_context"] for sample_id in selected_ids
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard/context-preserving 6.5k MC train subset")
    parser.add_argument("--source", type=Path, default=Path("datasets/derived/mc_bootstrap_v1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/derived/mc_bootstrap_v2_6500"))
    parser.add_argument(
        "--dms-root", type=Path, default=Path("datasets/raw/kaggle_habbas_dms_v1/1")
    )
    parser.add_argument("--target-train", type=int, default=TARGET_TRAIN)
    args = parser.parse_args()
    if args.target_train != TARGET_TRAIN:
        raise ValueError(f"this governed selection policy is fixed at {TARGET_TRAIN} train images")
    print(json.dumps(build_subset(args.source, args.output, args.dms_root, args.target_train), indent=2))


if __name__ == "__main__":
    main()
