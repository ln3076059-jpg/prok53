from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from training.common import sha256_file

IDENTITY_FIELDS = (
    "source_id",
    "camera_id",
    "video_id",
    "clip_id",
    "vehicle_id",
    "person_id",
)
GROUP_FIELDS = (
    "sha256",
    "phash",
    "source_group_id",
    "effective_group_id",
    "near_cluster_id",
    "video_id",
    "clip_id",
    "vehicle_id",
    "person_id",
    "camera_id",
)
CONDITION_ALIASES = {
    "day": {"day", "daylight", "daylight_or_well_lit"},
    "night": {"night"},
    "low_light": {"low_light"},
    "rain": {"rain"},
    "glare": {"glare", "windshield_glare"},
    "reflection": {"reflection", "windshield_reflection"},
    "blur": {"blur", "motion_blur"},
    "occlusion": {"occlusion", "partial_occlusion"},
    "dark_windshield": {"dark_windshield"},
    "camera_angle": {"camera_angle", "oblique_view", "different_camera_angle"},
}
UNKNOWN = "UNKNOWN"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _declared_values(item: dict, field: str) -> list[str]:
    value = item.get(field)
    if value in (None, "", []):
        return []
    values = value if isinstance(value, list) else [value]
    return [str(entry) for entry in values if str(entry).strip()]


def _field_report(items: list[dict], field: str) -> dict:
    values = [value for item in items for value in _declared_values(item, field)]
    present_samples = sum(bool(_declared_values(item, field)) for item in items)
    return {
        "status": "DECLARED" if present_samples == len(items) and items else UNKNOWN,
        "present_samples": present_samples,
        "missing_samples": len(items) - present_samples,
        "unique_values": len(set(values)),
        "distribution": dict(sorted(Counter(values).items())),
    }


def _image_index(root: Path) -> dict[tuple[str, str], Path]:
    index = {}
    image_root = root / "images"
    if not image_root.is_dir():
        return index
    for split_dir in image_root.iterdir():
        if split_dir.is_dir():
            for path in split_dir.iterdir():
                if path.is_file():
                    index[(split_dir.name, path.stem)] = path
    return index


def _resolve_image(root: Path, item: dict, image_index: dict[tuple[str, str], Path]) -> Path | None:
    for key in ("image_path", "path", "reviewed_path", "ingested_path"):
        if item.get(key) and Path(item[key]).is_file():
            return Path(item[key])
    split = item.get("split")
    sample_id = item.get("sample_id")
    if split and sample_id:
        return image_index.get((str(split), str(sample_id)))
    return None


def _resolution_report(root: Path, items: list[dict]) -> dict:
    resolutions: Counter[str] = Counter()
    unreadable = 0
    image_index = _image_index(root)
    for item in items:
        path = _resolve_image(root, item, image_index)
        if path is None:
            unreadable += 1
            continue
        try:
            with Image.open(path) as image:
                resolutions[f"{image.width}x{image.height}"] += 1
        except OSError:
            unreadable += 1
    return {
        "status": "MEASURED" if not unreadable and items else UNKNOWN,
        "measured_samples": sum(resolutions.values()),
        "unknown_samples": unreadable,
        "unique_resolutions": len(resolutions),
        "distribution": dict(sorted(resolutions.items())),
    }


def _condition_report(items: list[dict]) -> dict:
    declared = Counter(
        value.strip().lower() for item in items for value in _declared_values(item, "conditions")
    )
    coverage = {}
    for name, aliases in CONDITION_ALIASES.items():
        count = sum(declared[alias] for alias in aliases)
        coverage[name] = {"status": "DECLARED" if count else UNKNOWN, "samples": count}
    return {
        "samples_with_declared_conditions": sum(
            bool(_declared_values(item, "conditions")) for item in items
        ),
        "unknown_samples": sum(not _declared_values(item, "conditions") for item in items),
        "declared_distribution": dict(sorted(declared.items())),
        "requested_coverage": coverage,
    }


def _leakage_report(items: list[dict]) -> dict:
    report = {}
    for field in GROUP_FIELDS:
        split_by_value: defaultdict[str, set[str]] = defaultdict(set)
        present = 0
        for item in items:
            values = _declared_values(item, field)
            if values and item.get("split"):
                present += 1
                for value in values:
                    split_by_value[value].add(str(item["split"]))
        overlaps = sorted(value for value, splits in split_by_value.items() if len(splits) > 1)
        if overlaps:
            status = "FAIL"
        elif present == len(items) and items:
            status = "PASS"
        else:
            status = "NOT_PROVABLE"
        report[field] = {
            "status": status,
            "present_samples": present,
            "unique_values": len(split_by_value),
            "cross_split_overlap_count": len(overlaps),
            "cross_split_overlap_examples": overlaps[:20],
        }
    return report


def audit_dataset(name: str, root: Path, manifest_path: Path) -> dict:
    items = _read_jsonl(manifest_path)
    statuses = Counter(
        str(item.get("human_review_status") or item.get("review_status") or UNKNOWN)
        for item in items
    )
    governed = bool(items) and set(statuses).issubset({"APPROVED", "APPROVED_NEGATIVE"})
    fields = {field: _field_report(items, field) for field in IDENTITY_FIELDS}
    fields["vehicle_type"] = _field_report(items, "vehicle_type")
    fields["occupant_role"] = _field_report(items, "occupant_role")
    leakage = _leakage_report(items)
    source_groups = _field_report(items, "source_group_id")
    return {
        "name": name,
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "samples": len(items),
        "governance_status": "GOVERNED" if governed else "NOT_GOVERNED",
        "human_review_statuses": dict(sorted(statuses.items())),
        "splits": dict(sorted(Counter(str(item.get("split", UNKNOWN)) for item in items).items())),
        "class_distribution": dict(
            sorted(
                Counter(str(item["class_name"]) for item in items if item.get("class_name")).items()
            )
        ),
        "metadata": fields,
        "conditions": _condition_report(items),
        "resolution": _resolution_report(root, items),
        "group_diversity": {
            "samples": len(items),
            "declared_source_groups": source_groups["unique_values"],
            "provider_augmentations_count_as_new_diversity": False,
        },
        "leakage": leakage,
        "subject_disjoint_status": leakage["person_id"]["status"],
    }


def build_report(specs: list[tuple[str, Path, Path]]) -> dict:
    datasets = [audit_dataset(name, root, manifest) for name, root, manifest in specs]
    return {
        "schema_version": 1,
        "status": "NOT_GOVERNED"
        if any(item["governance_status"] != "GOVERNED" for item in datasets)
        else "GOVERNED",
        "datasets": datasets,
        "scientific_claims": {
            "subject_disjoint": "NOT_PROVABLE"
            if any(item["subject_disjoint_status"] != "PASS" for item in datasets)
            else "PROVABLE",
            "external_domain_generalization": "NOT_RUN",
            "human_approved_ground_truth": all(
                item["governance_status"] == "GOVERNED" for item in datasets
            ),
        },
        "policy": (
            "UNKNOWN means the manifest does not declare enough metadata. Image resolution may be "
            "measured from files; semantic conditions and identities are never inferred."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# V2 Data Diversity Audit",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "This report uses declared manifest metadata and measured image dimensions only. Missing "
        "identity or condition metadata is `UNKNOWN`; subject separation is not inferred.",
        "",
    ]
    for dataset in report["datasets"]:
        top_resolutions = sorted(
            dataset["resolution"]["distribution"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        lines.extend(
            [
                f"## {dataset['name']}",
                "",
                f"- Samples: **{dataset['samples']}**",
                f"- Governance: **{dataset['governance_status']}**",
                f"- Human review states: `{dataset['human_review_statuses']}`",
                f"- Splits: `{dataset['splits']}`",
                f"- Source/provider: `{dataset['metadata']['source_id']}`",
                "- Declared source groups: "
                f"**{dataset['group_diversity']['declared_source_groups']}**",
                f"- Resolution status: **{dataset['resolution']['status']}**; "
                f"measured {dataset['resolution']['measured_samples']}, unknown "
                f"{dataset['resolution']['unknown_samples']}, unique "
                f"{dataset['resolution']['unique_resolutions']}",
                f"- Top measured resolutions: `{dict(top_resolutions)}`",
                f"- Subject-disjoint: **{dataset['subject_disjoint_status']}**",
                "",
                "### Identity and semantic metadata",
                "",
                "| Dimension | Status | Present | Missing | Unique |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for field, values in dataset["metadata"].items():
            lines.append(
                f"| {field} | {values['status']} | {values['present_samples']} | "
                f"{values['missing_samples']} | {values['unique_values']} |"
            )
        lines.extend(
            [
                "",
                "### Requested condition coverage",
                "",
                "| Condition | Status | Declared samples |",
                "|---|---:|---:|",
            ]
        )
        for condition, values in dataset["conditions"]["requested_coverage"].items():
            lines.append(f"| {condition} | {values['status']} | {values['samples']} |")
        lines.extend(
            [
                "",
                "### Leakage evidence",
                "",
                "| Dimension | Status | Present | Unique | Cross-split overlaps |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for field, values in dataset["leakage"].items():
            lines.append(
                f"| {field} | {values['status']} | {values['present_samples']} | "
                f"{values['unique_values']} | {values['cross_split_overlap_count']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Scientific claim boundary",
            "",
            f"- Subject-disjoint: **{report['scientific_claims']['subject_disjoint']}**",
            "- External-domain generalization: **NOT_RUN**",
            "- Human-approved ground truth: "
            f"**{report['scientific_claims']['human_approved_ground_truth']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit actual V2 data diversity and leakage evidence"
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/V2_DATA_DIVERSITY_AUDIT.json")
    )
    parser.add_argument(
        "--output-md", type=Path, default=Path("reports/V2_DATA_DIVERSITY_AUDIT.md")
    )
    args = parser.parse_args()
    specs = [
        (
            "PHONE_DETECTOR_PROPOSAL_DATA",
            Path("datasets/derived/phone_bootstrap_v2"),
            Path("datasets/derived/phone_bootstrap_v2/manifest.jsonl"),
        ),
        (
            "UPPER_BODY_ROI_PROPOSAL_DATA",
            Path("datasets/derived/seatbelt_v2_balanced"),
            Path("datasets/derived/seatbelt_v2_balanced/manifest.jsonl"),
        ),
        (
            "SEATBELT_CLASSIFIER_PROPOSAL_DATA",
            Path("datasets/derived/seatbelt_classifier_v2"),
            Path("datasets/derived/seatbelt_classifier_v2/manifest.jsonl"),
        ),
    ]
    report = build_report(specs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "datasets": len(report["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
