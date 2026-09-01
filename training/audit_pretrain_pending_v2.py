from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from training.common import read_labels, sha256_file, stable_json_hash

SPLITS = ("train", "val", "test")
EXTERNAL_ONLY_SOURCES = {
    "mendeley_driver_risk_behavior_v1",
    "roboflow_adt_seatbelt_v1",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _group_overlap(items: list[dict]) -> dict:
    by_group: dict[str, set[str]] = {}
    for item in items:
        by_group.setdefault(str(item["effective_group_id"]), set()).add(item["split"])
    leaking = sorted(group for group, splits in by_group.items() if len(splits) > 1)
    return {
        "groups": len(by_group),
        "leaking_groups": leaking,
        "status": "PASS" if not leaking else "FAIL",
    }


def _detector(root: Path, allowed_classes: set[int]) -> tuple[dict, list[str]]:
    errors = []
    items = _jsonl(root / "manifest.jsonl")
    by_id = {item["sample_id"]: item for item in items}
    if len(by_id) != len(items):
        errors.append(f"{root}: duplicate sample_id")
    counts = {}
    box_counts = {}
    for split in SPLITS:
        images = [path for path in (root / "images" / split).glob("*") if path.is_file()]
        labels = list((root / "labels" / split).glob("*.txt"))
        expected = [item for item in items if item["split"] == split]
        counts[split] = len(images)
        if len(images) != len(labels) or len(images) != len(expected):
            errors.append(f"{root}/{split}: image/label/manifest count mismatch")
        boxes = 0
        for label_path in labels:
            for label in read_labels(label_path):
                if label.class_id not in allowed_classes:
                    errors.append(f"{label_path}: unexpected class {label.class_id}")
                boxes += 1
        box_counts[split] = boxes
    for item in items:
        if item.get("usage") != "MODEL_ASSISTED_PENDING_APPROVAL":
            errors.append(f"{item['sample_id']}: incorrect usage policy")
        if item.get("reviewer_type") != "AUTOMATED" or item.get("governance_eligible") is not False:
            errors.append(f"{item['sample_id']}: incorrect automated-review governance provenance")
        if item.get("source_id") in EXTERNAL_ONLY_SOURCES:
            errors.append(f"{item['sample_id']}: external-only source admitted to training")
    total_boxes = sum(box_counts.values())
    if total_boxes:
        for split in ("val", "test"):
            positive_share = box_counts[split] / total_boxes
            if positive_share < 0.05:
                errors.append(
                    f"{root}/{split}: positive box share {positive_share:.3f} is below 0.05"
                )
    isolation = _group_overlap(items)
    if isolation["status"] != "PASS":
        errors.append(f"{root}: effective group leakage")
    return {
        "samples": len(items),
        "images": counts,
        "boxes": box_counts,
        "manifest_sha256": stable_json_hash(items),
        "isolation": isolation,
    }, errors


def _classifier(root: Path) -> tuple[dict, list[str]]:
    errors = []
    items = _jsonl(root / "manifest.jsonl")
    counts = {split: Counter() for split in SPLITS}
    classes_by_hash: dict[str, set[str]] = {}
    classes_by_source_box: dict[tuple[str, int], set[str]] = {}
    for item in items:
        counts[item["split"]][item["class_name"]] += 1
        classes_by_hash.setdefault(str(item["sha256"]), set()).add(item["class_name"])
        source_box = (str(item["source_sample_id"]), int(item.get("source_box_index", 0)))
        classes_by_source_box.setdefault(source_box, set()).add(item["class_name"])
        if item.get("reviewer_type") != "AUTOMATED" or item.get("governance_eligible") is not False:
            errors.append(f"{item['sample_id']}: incorrect classifier automated-review provenance")
        path = Path(item["image_path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            errors.append(f"{item['sample_id']}: classifier crop missing or hash mismatch")
    report = _json(root / "CLASSIFIER_DATASET_REPORT.json")
    manifest_hash = stable_json_hash(items)
    if report.get("manifest_sha256") != manifest_hash:
        errors.append("classifier: dataset report manifest hash is stale")
    expected_names = {"seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded"}
    for split in SPLITS:
        missing_class = set(counts[split]) != expected_names or any(
            counts[split][name] < 1 for name in expected_names
        )
        if missing_class:
            errors.append(f"classifier {split}: missing one or more classes")
    isolation = _group_overlap(items)
    if isolation["status"] != "PASS":
        errors.append("classifier: effective group leakage")
    hash_conflicts = sorted(key for key, value in classes_by_hash.items() if len(value) > 1)
    source_box_conflicts = sorted(
        f"{key[0]}:box:{key[1]}" for key, value in classes_by_source_box.items() if len(value) > 1
    )
    if hash_conflicts:
        errors.append(f"classifier: {len(hash_conflicts)} cross-class image hash conflicts")
    if source_box_conflicts:
        errors.append(f"classifier: {len(source_box_conflicts)} cross-class source ROI conflicts")
    if report.get("ready_for_training") != report.get("ready_for_exploratory_training"):
        errors.append("classifier exploratory readiness flags are inconsistent")
    if report.get("ready_for_governed_training"):
        errors.append("classifier must not claim governed readiness")
    return {
        "samples": len(items),
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "manifest_sha256": manifest_hash,
        "isolation": isolation,
        "ready_for_exploratory_training": bool(report.get("ready_for_exploratory_training")),
        "hash_conflicts": hash_conflicts,
        "source_box_conflicts": source_box_conflicts,
    }, errors


def _weight(path: Path, provenance_path: Path) -> tuple[dict, list[str]]:
    errors = []
    provenance = _json(provenance_path)
    actual_hash = sha256_file(path) if path.is_file() else None
    actual_bytes = path.stat().st_size if path.is_file() else None
    if actual_hash != provenance.get("sha256") or actual_bytes != provenance.get("bytes"):
        errors.append(f"base weight provenance mismatch: {path}")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_hash}, errors


def audit(
    root: Path = Path("datasets/derived/v2_pretrain_pending_approval"),
    config_path: Path = Path("experiments/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL/config.yaml"),
) -> dict:
    phone, phone_errors = _detector(root / "phone_detector", {0})
    seatbelt, seatbelt_errors = _detector(root / "seatbelt_detector", {0})
    classifier, classifier_errors = _classifier(root / "seatbelt_classifier")
    detector_weight, detector_weight_errors = _weight(
        Path("models/base/yolo11s.pt"), Path("models/base/yolo11s.provenance.json")
    )
    classifier_weight, classifier_weight_errors = _weight(
        Path("models/base/yolo11s-cls.pt"),
        Path("models/base/yolo11s-cls.provenance.json"),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_errors = []
    if config.get("governance_level") != "MODEL_ASSISTED_PENDING_APPROVAL":
        config_errors.append(
            "experiment config lacks MODEL_ASSISTED_PENDING_APPROVAL governance level"
        )
    if any(
        component.get("multi_scale") is not False
        for component in config.get("components", {}).values()
        if component.get("task") == "detect"
    ):
        config_errors.append("detector multi_scale must remain disabled")
    dataset_report_errors = []
    dataset_report_path = root / "PRETRAIN_PENDING_APPROVAL_DATASET_REPORT.json"
    if not dataset_report_path.is_file():
        dataset_report_errors.append("pretrain dataset report is missing")
    else:
        dataset_report = _json(dataset_report_path)
        computed = {
            "phone_detector": phone,
            "seatbelt_detector": seatbelt,
            "seatbelt_classifier": classifier,
        }
        for name, payload in computed.items():
            recorded = dataset_report.get("datasets", {}).get(name, {})
            if recorded.get("manifest_sha256") != payload["manifest_sha256"]:
                dataset_report_errors.append(
                    f"pretrain dataset report has stale {name} manifest hash"
                )
        if dataset_report.get("governance_level") != "MODEL_ASSISTED_PENDING_APPROVAL":
            dataset_report_errors.append("pretrain dataset report governance level is invalid")
        if bool(dataset_report.get("training_ready")) != bool(
            classifier["ready_for_exploratory_training"]
        ):
            dataset_report_errors.append("pretrain dataset report readiness flag is stale")
        if dataset_report.get("governed_training_ready") is not False:
            dataset_report_errors.append(
                "pretrain dataset report must not claim governed readiness"
            )
    errors = [
        *phone_errors,
        *seatbelt_errors,
        *classifier_errors,
        *detector_weight_errors,
        *classifier_weight_errors,
        *config_errors,
        *dataset_report_errors,
    ]
    integrity_pass = not errors
    proposal_detector_ready = integrity_pass
    exploratory_ready = integrity_pass and classifier["ready_for_exploratory_training"]
    return {
        "schema_version": 1,
        "status": "PASS" if integrity_pass else "FAIL",
        "readiness": {
            "proposal_detector_training_ready": proposal_detector_ready,
            "classifier_training_ready": bool(classifier["ready_for_exploratory_training"]),
            "exploratory_training_ready": exploratory_ready,
            "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
            "human_verified": False,
            "governed_training_ready": False,
            "production_activation_ready": False,
        },
        "datasets": {
            "phone_detector": phone,
            "seatbelt_detector": seatbelt,
            "seatbelt_classifier": classifier,
        },
        "base_weights": {
            "detector": detector_weight,
            "classifier": classifier_weight,
        },
        "errors": errors,
        "training_blockers": (
            []
            if exploratory_ready
            else ["CLASSIFIER_UNCERTAIN_EVIDENCE_BELOW_MINIMUM"]
            if integrity_pass
            else ["DATASET_INTEGRITY_AUDIT_FAILED"]
        ),
        "policy": (
            "PASS means dataset integrity passed. Training requires the separate readiness flag. "
            "No result here means HUMAN_APPROVED, governed, production-ready, or externally tested."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model-assisted pending-approval exploratory V2 datasets"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/derived/v2_pretrain_pending_approval")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL/config.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2_pretrain_pending_approval_readiness.json")
    )
    args = parser.parse_args()
    report = audit(args.root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
