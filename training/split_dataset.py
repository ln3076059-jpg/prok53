# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from training.common import stable_json_hash
from training.data_governance import PROTECTED_SUBJECTS, validate_group_split

LINK_DIMENSIONS = (
    "sha256",
    "near_cluster_id",
    "trusted_subject_id",
    "video_id",
    "session_id",
    "original_asset_id",
)


def _linked_group_components(grouped: dict[str, list[dict]]) -> dict[str, str]:
    """Join groups that share a subject, video, source asset, or duplicate identity."""
    parent = {group: group for group in grouped}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[tuple[str, str], str] = {}
    for group, items in grouped.items():
        for item in items:
            for dimension in LINK_DIMENSIONS:
                value = item.get(dimension)
                if not value:
                    continue
                key = (dimension, str(value))
                if key in owners:
                    union(group, owners[key])
                else:
                    owners[key] = group
    return {group: find(group) for group in grouped}


def assign_groups(records: list[dict], seed: int = 42) -> dict[str, str]:
    """Keep linked subjects/sessions/original assets intact while balancing splits."""
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("review_status") != "APPROVED":
            continue
        group = record.get("effective_group_id") or record.get("source_group_id")
        if not group:
            raise ValueError(f"record {record.get('sample_id')} has no effective group")
        if record.get("trusted_subject_id") in PROTECTED_SUBJECTS:
            raise ValueError(
                f"protected subject cannot be materialized: {record['trusted_subject_id']}"
            )
        grouped[group].append(record)
    component_for = _linked_group_components(grouped)
    components: defaultdict[str, list[dict]] = defaultdict(list)
    component_groups: defaultdict[str, list[str]] = defaultdict(list)
    for group, items in grouped.items():
        component = component_for[group]
        components[component].extend(items)
        component_groups[component].append(group)
    rng = random.Random(seed)
    ordered = list(components.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    total = sum(len(items) for _, items in ordered)
    counts = Counter()
    class_totals = Counter()
    component_class_counts: dict[str, Counter] = {}
    for component, items in ordered:
        component_counts = Counter()
        for item in items:
            component_counts.update(
                {int(key): int(value) for key, value in item.get("class_counts", {}).items()}
            )
        component_class_counts[component] = component_counts
        class_totals.update(component_counts)
    split_class_counts: defaultdict[str, Counter] = defaultdict(Counter)
    assignments: dict[str, str] = {}
    for component, items in ordered:
        def deficit(split_name: str) -> float:
            sample_score = (targets[split_name] * total - counts[split_name]) / max(total, 1)
            class_score = sum(
                (targets[split_name] * class_totals[class_id] - split_class_counts[split_name][class_id])
                / max(class_totals[class_id], 1)
                for class_id in class_totals
            )
            return sample_score + class_score

        split = max(targets, key=deficit)
        for group in component_groups[component]:
            assignments[group] = split
        counts[split] += len(items)
        split_class_counts[split].update(component_class_counts[component])
    return assignments


def materialize(records: list[dict], assignments: dict[str, str], output: Path) -> list[dict]:
    manifest: list[dict] = []
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.get("review_status") != "APPROVED":
            continue
        group = record.get("effective_group_id") or record["source_group_id"]
        split = assignments[group]
        image_source = Path(record["reviewed_path"])
        label_source = Path(record["label_path"])
        image_target = output / "images" / split / f"{record['sample_id']}{image_source.suffix.lower()}"
        label_target = output / "labels" / split / f"{record['sample_id']}.txt"
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        item = {**record, "split": split, "effective_group_id": group}
        manifest.append(item)
    (output / "manifest.jsonl").write_text("\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n")
    return manifest


def validate_isolation(manifest: list[dict]) -> dict:
    dimensions = [
        "sha256",
        "near_cluster_id",
        "effective_group_id",
        "source_group_id",
        "video_id",
        "session_id",
        "original_asset_id",
    ]
    overlaps = {}
    for dimension in dimensions:
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            if item.get(dimension):
                seen[str(item[dimension])].add(item["split"])
        overlaps[dimension] = sorted(key for key, splits in seen.items() if len(splits) > 1)
    subject_values = [item.get("trusted_subject_id") for item in manifest]
    subject_status = "PASS" if subject_values and all(subject_values) else "NOT_PROVABLE"
    if subject_status == "PASS":
        seen_subjects: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            seen_subjects[item["trusted_subject_id"]].add(item["split"])
        if any(len(splits) > 1 for splits in seen_subjects.values()):
            subject_status = "FAIL"
    protected = sorted(
        {
            str(item.get("trusted_subject_id"))
            for item in manifest
            if item.get("trusted_subject_id") in PROTECTED_SUBJECTS
        }
    )
    status = (
        "PASS"
        if not any(overlaps.values()) and subject_status != "FAIL" and not protected
        else "FAIL"
    )
    return {
        "overlaps": overlaps,
        "subject_isolation": subject_status,
        "protected_subjects": protected,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset/governed/mc3_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.review_manifest.read_text().splitlines() if line]
    assignments = assign_groups(records, args.seed)
    proposed = []
    for record in records:
        if record.get("review_status") != "APPROVED":
            continue
        group = record.get("effective_group_id") or record["source_group_id"]
        proposed.append({**record, "effective_group_id": group, "split": assignments[group]})
    preflight = validate_group_split(proposed)
    if preflight["status"] != "PASS":
        raise ValueError(f"split preflight failed before materialization: {preflight}")
    manifest = materialize(records, assignments, args.output)
    isolation = validate_isolation(manifest)
    metadata = {"seed": args.seed, "ratios": {"train": 0.70, "val": 0.15, "test": 0.15}, "assignments": assignments, "isolation": isolation, "manifest_sha256": stable_json_hash(manifest)}
    (args.output / "split_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (args.output / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: phone\n  1: seatbelt_fastened\n  2: seatbelt_unfastened\nnc: 3\n")
    print(json.dumps(metadata, indent=2))
    raise SystemExit(0 if isolation["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
