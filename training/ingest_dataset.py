from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image

from training.common import perceptual_hash, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_FRAME_STEM = re.compile(r"^(?P<clip>.+)-\d+_jpg\.rf\.[^.]+$")
SEQUENCE_FRAME_STEM = re.compile(r"^(?P<clip>[0-9a-fA-F-]{36})_\d+$")
ROBOFLOW_ASSET_STEM = re.compile(r"^(?P<clip>.+?)(?:_a)?_jpg\.rf\.[^.]+$")


def source_label_path(image_path: Path, source_dir: Path) -> Path:
    """Locate a source label without mutating the immutable raw dataset."""
    adjacent = image_path.with_suffix(".txt")
    if adjacent.exists():
        return adjacent
    relative = image_path.relative_to(source_dir)
    if "images" in relative.parts:
        index = relative.parts.index("images")
        mapped = Path(*relative.parts[:index], "labels", *relative.parts[index + 1 :])
        return (source_dir / mapped).with_suffix(".txt")
    return adjacent


def source_group_id(image_path: Path, group_prefix: str, grouping_regex: str | None = None) -> str:
    match = VIDEO_FRAME_STEM.match(image_path.stem) or SEQUENCE_FRAME_STEM.match(image_path.stem)
    if match:
        return f"{group_prefix}:{match.group('clip')}"
    if grouping_regex:
        custom_match = re.match(grouping_regex, image_path.stem)
        if custom_match:
            return f"{group_prefix}:{custom_match.group('clip')}"
    asset_match = ROBOFLOW_ASSET_STEM.match(image_path.stem)
    if asset_match:
        return f"{group_prefix}:GROUP_REVIEW_REQUIRED:{asset_match.group('clip')}"
    return f"{group_prefix}:GROUP_REVIEW_REQUIRED:{image_path.stem}"


def read_source_labels(path: Path) -> list[tuple[int, list[float]]]:
    """Read raw source YOLO labels without assuming the canonical class vocabulary."""
    labels: list[tuple[int, list[float]]] = []
    if not path.exists():
        return labels
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid raw source label {path}:{line_number}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id < 0 or any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"invalid raw source label {path}:{line_number}")
        labels.append((class_id, values))
    return labels


def load_source(source_id: str, registry: Path) -> dict:
    document = yaml.safe_load(registry.read_text())
    match = next((item for item in document["sources"] if item["source_id"] == source_id), None)
    if not match:
        raise ValueError(f"unknown source_id: {source_id}")
    if match["status"].startswith("REJECTED") or match["license"] in {None, "UNKNOWN"}:
        raise ValueError(f"source rejected by license gate: {match['reason']}")
    return match


def ingest(source_dir: Path, destination: Path, source: dict, group_prefix: str) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen_sha: dict[str, str] = {}
    for image_path in sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        digest = sha256_file(image_path)
        if digest in seen_sha:
            records.append({"source_asset_id": str(image_path.relative_to(source_dir)), "status": "EXACT_DUPLICATE", "duplicate_of": seen_sha[digest], "sha256": digest})
            continue
        with Image.open(image_path) as image:
            image.verify()
        sample_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{source['source_id']}:{image_path.relative_to(source_dir)}:{digest}").hex
        target = destination / f"{sample_id}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target)
        label_path = source_label_path(image_path, source_dir)
        source_label_errors: list[str] = []
        try:
            labels = read_source_labels(label_path)
        except ValueError as exc:
            labels = []
            source_label_errors.append(str(exc))
        record = {
            "sample_id": sample_id,
            "source_id": source["source_id"],
            "source_asset_id": str(image_path.relative_to(source_dir)),
            "source_label_path": str(label_path.resolve()) if label_path.exists() else None,
            "source_group_id": source_group_id(image_path, group_prefix, source.get("grouping_regex")),
            "source_group_status": "INFERRED_SEQUENCE" if (
                VIDEO_FRAME_STEM.match(image_path.stem)
                or SEQUENCE_FRAME_STEM.match(image_path.stem)
                or (source.get("grouping_regex") and re.match(source["grouping_regex"], image_path.stem))
            ) else "GROUP_REVIEW_REQUIRED",
            "vehicle_context_required": True,
            "vehicle_context_id": None,
            "original_path": str(image_path.resolve()),
            "ingested_path": str(target.resolve()),
            "sha256": digest,
            "phash": perceptual_hash(image_path),
            "license": source["license"],
            "annotation_provenance": "source_import",
            "annotation_count": len(labels),
            "source_label_errors": source_label_errors,
            "source_image_class_hint": source.get("folder_classes", {}).get(image_path.parent.name),
            "review_status": "PENDING",
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        records.append(record)
        seen_sha[digest] = sample_id
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable, license-gated dataset ingestion")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--registry", type=Path, default=Path("datasets/sources.yaml"))
    parser.add_argument("--destination", type=Path, default=Path("datasets/incoming"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/ingest.jsonl"))
    args = parser.parse_args()
    if args.destination.resolve() == Path("datasets/raw").resolve():
        raise ValueError("datasets/raw is immutable and cannot be an ingestion destination")
    source = load_source(args.source_id, args.registry)
    records = ingest(args.source_dir, args.destination, source, args.group_prefix)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n")
    print(json.dumps({"records": len(records), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
