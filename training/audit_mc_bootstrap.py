from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from training.build_mc_bootstrap import CLASS_NAMES, SPLITS, STATUS, multi_index_near_pairs
from training.common import parse_yolo_line, sha256_file, stable_json_hash


def audit(root: Path, near_threshold: int = 4) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = {int(key): value for key, value in data.get("names", {}).items()}
    if names != CLASS_NAMES or data.get("nc") != 3:
        errors.append(f"canonical class schema mismatch: {names}")
    bootstrap = json.loads((root / "BOOTSTRAP_MANIFEST.json").read_text(encoding="utf-8"))
    if bootstrap.get("status") != STATUS:
        errors.append("dataset is not marked proposal-model-only")

    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_sample = {item["sample_id"]: item for item in manifest}
    if len(by_sample) != len(manifest):
        errors.append("duplicate sample_id values in manifest")
    if any(item.get("usage") != "PROPOSAL_MODEL_ONLY" for item in manifest):
        errors.append("manifest contains non-proposal usage")
    if any(item.get("human_review_status") != "PENDING" for item in manifest):
        errors.append("bootstrap manifest contains non-pending human review state")

    dimensions = ("sha256", "base_group_id", "effective_group_id", "near_cluster_id")
    overlaps = {}
    for dimension in dimensions:
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            seen[str(item[dimension])].add(item["split"])
        overlaps[dimension] = sorted(value for value, splits in seen.items() if len(splits) > 1)
        if overlaps[dimension]:
            errors.append(f"{dimension} crosses splits: {len(overlaps[dimension])}")

    split_report = {}
    manifest_counts = Counter(item["split"] for item in manifest)
    for split in SPLITS:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.is_file())
        labels = sorted(path for path in label_dir.glob("*.txt") if path.is_file())
        if len(images) != len(labels) or len(images) != manifest_counts[split]:
            errors.append(
                f"{split} file/manifest count mismatch: {len(images)}/{len(labels)}/{manifest_counts[split]}"
            )
        instances = Counter()
        positives = Counter()
        for image_path in images:
            item = by_sample.get(image_path.stem)
            if item is None or item.get("split") != split:
                errors.append(f"missing/wrong manifest record: {split}/{image_path.name}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"corrupt image {image_path}: {exc}")
                continue
            if sha256_file(image_path) != item["sha256"]:
                errors.append(f"image SHA-256 mismatch: {split}/{image_path.name}")
            label_path = label_dir / f"{image_path.stem}.txt"
            seen_lines = set()
            sample_classes = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                if line in seen_lines:
                    errors.append(f"duplicate label line: {label_path}:{line_number}")
                seen_lines.add(line)
                try:
                    parsed = parse_yolo_line(line)
                except ValueError as exc:
                    errors.append(f"invalid label {label_path}:{line_number}: {exc}")
                    continue
                instances[parsed.class_id] += 1
                sample_classes.add(parsed.class_id)
            positives.update(sample_classes)
        if any(instances[class_id] == 0 for class_id in CLASS_NAMES):
            errors.append(f"{split} is missing at least one canonical class")
        split_report[split] = {
            "images": len(images),
            "instances": {CLASS_NAMES[key]: instances[key] for key in CLASS_NAMES},
            "positive_images": {CLASS_NAMES[key]: positives[key] for key in CLASS_NAMES},
        }

    values = [int(item["phash"], 16) for item in manifest]
    near_pairs = multi_index_near_pairs(values, near_threshold)
    cross_split_near = [
        [manifest[left]["sample_id"], manifest[right]["sample_id"]]
        for left, right in near_pairs
        if manifest[left]["split"] != manifest[right]["split"]
    ]
    if cross_split_near:
        errors.append(f"near-duplicate pairs cross splits: {len(cross_split_near)}")
    if bootstrap.get("component_overlap") and any(bootstrap["component_overlap"].values()):
        errors.append("bootstrap summary reports component overlap")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset_status": bootstrap.get("status"),
        "manifest_samples": len(manifest),
        "manifest_sha256": stable_json_hash(manifest),
        "near_threshold": near_threshold,
        "near_pairs": len(near_pairs),
        "cross_split_near_pairs": cross_split_near,
        "overlaps": overlaps,
        "splits": split_report,
        "errors": errors,
        "warnings": warnings,
        "policy": "PASS validates bootstrap integrity and isolation only; it does not mean human-approved ground truth.",
    }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# MC bootstrap v1 data quality",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Dataset status: `{report['dataset_status']}`",
        "",
        f"Manifest samples: {report['manifest_samples']}",
        f"Near pairs at Hamming ≤ {report['near_threshold']}: {report['near_pairs']}",
        f"Cross-split near pairs: {len(report['cross_split_near_pairs'])}",
        "",
        "## Splits",
        "",
    ]
    for split, values in report["splits"].items():
        lines.append(f"- `{split}`: {values}")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {value}" for value in report["errors"] or ["None"])
    lines.extend(
        [
            "",
            "PASS validates file integrity and leakage isolation only. All labels remain proposal-only and require human review before final MC_001 training or accuracy claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the three-class proposal bootstrap dataset")
    parser.add_argument("root", type=Path, nargs="?", default=Path("datasets/derived/mc_bootstrap_v1"))
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/mc_bootstrap_v1"))
    args = parser.parse_args()
    report = audit(args.root, args.near_threshold)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.report_dir / "data_quality.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cross_split_near_pairs"}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
