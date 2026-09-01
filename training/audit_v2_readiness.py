from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from training.common import sha256_file

DIVERSITY_FIELDS = ("video_id", "vehicle_id", "person_id", "camera_id")
REQUIRED_CONDITIONS = {
    "daylight",
    "low_light",
    "windshield_reflection",
    "motion_blur",
    "video_compression",
    "partial_occlusion",
    "oblique_view",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _distribution(items: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key, "MISSING")) for item in items).items()))


def _dataset_summary(items: list[dict], box_key: str) -> dict:
    split_sources: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    conditions: set[str] = set()
    for item in items:
        split = str(item.get("split"))
        if split in split_sources and item.get("source_id"):
            split_sources[split].add(str(item["source_id"]))
        value = item.get("conditions", [])
        if isinstance(value, str):
            conditions.add(value)
        elif isinstance(value, list):
            conditions.update(str(entry) for entry in value)
    train_sources = split_sources["train"]
    test_sources = split_sources["test"]
    external_test = bool(test_sources) and train_sources.isdisjoint(test_sources)
    diversity = {
        field: {
            "present_samples": sum(bool(item.get(field)) for item in items),
            "unique_values": len({str(item[field]) for item in items if item.get(field)}),
        }
        for field in DIVERSITY_FIELDS
    }
    box_counts = [int(item.get(box_key, 0)) for item in items]
    return {
        "samples": len(items),
        "sources": _distribution(items, "source_id"),
        "review_statuses": _distribution(items, "human_review_status"),
        "usage": _distribution(items, "usage"),
        "group_review_required": sum(
            "GROUP_REVIEW_REQUIRED" in str(item.get("source_group_id", "")) for item in items
        ),
        "box_instances": sum(box_counts),
        "zero_box_samples": sum(count == 0 for count in box_counts),
        "diversity_fields": diversity,
        "condition_coverage": sorted(conditions),
        "missing_required_conditions": sorted(REQUIRED_CONDITIONS - conditions),
        "split_sources": {split: sorted(values) for split, values in split_sources.items()},
        "external_test_source_disjoint": external_test,
    }


def _model_placeholders(model_config: Path) -> list[str]:
    payload = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    markers = ("UNTRAINED", "PLACEHOLDER", "PENDING")
    return [marker for marker in markers if marker in serialized]


def audit(
    phone_dataset: Path = Path("datasets/derived/phone_bootstrap_v2"),
    seatbelt_dataset: Path = Path("datasets/derived/seatbelt_v2_balanced"),
    classifier_dataset: Path = Path("datasets/derived/seatbelt_classifier_v2"),
    phone_audit_path: Path = Path("reports/phone_bootstrap_v2/data_quality.json"),
    seatbelt_audit_path: Path = Path("reports/seatbelt_v2/audit.json"),
    model_config: Path = Path("models/model_config_v2.yaml"),
) -> dict:
    phone_items = _jsonl(phone_dataset / "manifest.jsonl")
    seatbelt_items = _jsonl(seatbelt_dataset / "manifest.jsonl")
    phone = _dataset_summary(phone_items, "phone_boxes")
    seatbelt_with_counts = [
        {**item, "instance_count": sum(item.get("class_counts", {}).values())}
        for item in seatbelt_items
    ]
    seatbelt = _dataset_summary(seatbelt_with_counts, "instance_count")
    classifier = _json(classifier_dataset / "CLASSIFIER_DATASET_REPORT.json")
    phone_integrity = _json(phone_audit_path)
    seatbelt_integrity = _json(seatbelt_audit_path)
    base_weight = Path("models/base/yolo11s.pt")
    base_provenance = _json(Path("models/base/yolo11s.provenance.json"))
    base_weight_valid = (
        base_weight.is_file()
        and base_weight.stat().st_size == int(base_provenance.get("bytes", -1))
        and sha256_file(base_weight) == base_provenance.get("sha256")
    )
    adt_report_path = Path("reports/adt_seatbelt_v1/review_source_audit.json")
    adt_candidate = _json(adt_report_path) if adt_report_path.is_file() else None
    gap_plan_path = Path("reports/v2_data_gap_action_plan.json")
    gap_plan = _json(gap_plan_path) if gap_plan_path.is_file() else None
    gap_external = gap_plan.get("external_candidates", {}) if gap_plan else {}

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def block(code: str, area: str, message: str, remediation: str) -> None:
        blockers.append(
            {
                "code": code,
                "severity": "BLOCKER",
                "area": area,
                "message": message,
                "remediation": remediation,
            }
        )

    def warn(code: str, area: str, message: str, remediation: str) -> None:
        warnings.append(
            {
                "code": code,
                "severity": "WARNING",
                "area": area,
                "message": message,
                "remediation": remediation,
            }
        )

    for name, summary in (("phone", phone), ("seatbelt", seatbelt)):
        pending = int(summary["review_statuses"].get("PENDING", 0))
        if pending:
            block(
                f"{name.upper()}_SEMANTIC_REVIEW_PENDING",
                name,
                f"{pending} samples are still PENDING.",
                "Apply append-only human review decisions; never promote proposals automatically.",
            )
        group_pending = int(summary["group_review_required"])
        if group_pending:
            block(
                f"{name.upper()}_SOURCE_GROUP_REVIEW_REQUIRED",
                name,
                f"{group_pending} samples still use GROUP_REVIEW_REQUIRED source groups.",
                "Resolve video/vehicle/person grouping before governed split certification.",
            )
        if len(summary["sources"]) < 2:
            block(
                f"{name.upper()}_SINGLE_SOURCE_DOMAIN",
                name,
                "All samples come from one provider/domain.",
                "Add a licensed, independently reviewed source with distinct cameras and vehicles.",
            )
        if not summary["external_test_source_disjoint"]:
            block(
                f"{name.upper()}_NO_EXTERNAL_TEST_DOMAIN",
                name,
                "The test split is not source-disjoint from training.",
                "Create a frozen external test set from an unseen provider/camera domain.",
            )
        missing_fields = [
            field
            for field, values in summary["diversity_fields"].items()
            if int(values["present_samples"]) != int(summary["samples"])
        ]
        if missing_fields:
            block(
                f"{name.upper()}_DIVERSITY_METADATA_INCOMPLETE",
                name,
                f"Required diversity fields are incomplete: {', '.join(missing_fields)}.",
                "Record real video, vehicle, person, and camera identifiers for every sample.",
            )
        if summary["missing_required_conditions"]:
            block(
                f"{name.upper()}_CONDITION_COVERAGE_UNPROVEN",
                name,
                "Required adverse-condition coverage is not declared in the manifest.",
                "Capture and review every condition defined by datasets/v2_capture_policy.yaml.",
            )

    if phone["zero_box_samples"]:
        block(
            "PHONE_NEGATIVES_NOT_HUMAN_CONFIRMED",
            "phone",
            f"{phone['zero_box_samples']} samples contain zero phone boxes but remain proposals.",
            "Review zero-box images so visible phones are not trained as background.",
        )
    if seatbelt["zero_box_samples"] == 0:
        block(
            "SEATBELT_ROI_HARD_NEGATIVES_MISSING",
            "seatbelt",
            "Every seatbelt-detector image contains an upper-body box.",
            "Add reviewed empty-cabin, seat, reflection, and non-occupant hard negatives.",
        )
    if not classifier.get("ready_for_training"):
        block(
            "SEATBELT_CLASSIFIER_NOT_READY",
            "classifier",
            "The three-state classifier is missing a reviewed class in one or more splits.",
            "Review independent UNCERTAIN_OR_OCCLUDED crops in train, val, and test.",
        )
    train_counts = classifier.get("counts", {}).get("train", {})
    fastened = int(train_counts.get("seatbelt_fastened", 0))
    unfastened = int(train_counts.get("seatbelt_unfastened", 0))
    if unfastened and fastened / unfastened > 1.5:
        warn(
            "SEATBELT_CLASSIFIER_TRAIN_IMBALANCE",
            "classifier",
            f"Fastened/unfastened train ratio is {fastened / unfastened:.2f}:1.",
            "Use reviewed class balancing and report per-class recall; do not duplicate test data.",
        )
    warn(
        "EXTERNAL_RESUME_BACKUP_REQUIRES_CONFIGURATION",
        "operations",
        "Automatic resume protects the working disk unless a second backup directory is supplied.",
        (
            "Pass -BackupDirectory on the portable launcher and place it on another disk "
            "or sync target."
        ),
    )

    placeholders = _model_placeholders(model_config)
    if placeholders:
        block(
            "PRODUCTION_CALIBRATION_INCOMPLETE",
            "deployment",
            f"Model configuration still contains markers: {', '.join(placeholders)}.",
            "Calibrate thresholds/camera ROIs, lock weights, then run frozen test once.",
        )

    proposal_ready = (
        phone_integrity.get("status") == "PASS"
        and seatbelt_integrity.get("status") == "PASS"
        and base_weight_valid
        and phone["samples"] > 0
        and seatbelt["samples"] > 0
    )
    governed_codes = {
        item["code"] for item in blockers if item["area"] in {"phone", "seatbelt", "classifier"}
    }
    governed_ready = proposal_ready and not governed_codes
    production_ready = governed_ready and not placeholders
    return {
        "schema_version": 1,
        "status": {
            "proposal_detector_training_ready": proposal_ready,
            "governed_training_ready": governed_ready,
            "production_activation_ready": production_ready,
        },
        "datasets": {"phone": phone, "seatbelt": seatbelt, "classifier": classifier},
        "integrity": {
            "phone": phone_integrity.get("status"),
            "seatbelt": seatbelt_integrity.get("status"),
            "base_weight_valid": base_weight_valid,
        },
        "model_config_markers": placeholders,
        "candidate_external_sources": {
            "mendeley_driver_risk_behavior_v1": gap_external.get(
                "mendeley_driver_risk_behavior_v1"
            ),
            "roboflow_adt_seatbelt_v1": adt_candidate,
        },
        "data_gap_action_plan": gap_plan,
        "blockers": blockers,
        "warnings": warnings,
        "blocker_codes": [item["code"] for item in blockers],
        "policy": (
            "Proposal readiness permits detector bootstrap training only. Governed and production "
            "readiness require real review, diversity, external testing, and calibration."
        ),
    }


def markdown(report: dict) -> str:
    status = report["status"]
    lines = [
        "# V2 training readiness",
        "",
        f"- Proposal detector training ready: `{status['proposal_detector_training_ready']}`",
        f"- Governed training ready: `{status['governed_training_ready']}`",
        f"- Production activation ready: `{status['production_activation_ready']}`",
        "",
        "## Blocking work",
        "",
    ]
    for item in report["blockers"]:
        lines.extend(
            [
                f"### {item['code']}",
                "",
                item["message"],
                "",
                f"Required action: {item['remediation']}",
                "",
            ]
        )
    lines.extend(["## Warnings", ""])
    for item in report["warnings"]:
        lines.extend(
            [
                f"### {item['code']}",
                "",
                item["message"],
                "",
                f"Recommended action: {item['remediation']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write(report: dict, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    return output, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V2 proposal/governed/production readiness")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v2_training_readiness.json"),
    )
    args = parser.parse_args()
    report = audit()
    write(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
