from __future__ import annotations

import json

from PIL import Image

from training.audit_v2_diversity import audit_dataset, build_report
from training.common import perceptual_hash, sha256_file


def test_diversity_audit_preserves_unknown_metadata_and_detects_group_leakage(tmp_path):
    root = tmp_path / "dataset"
    records = []
    for split, color in (("train", "red"), ("val", "blue")):
        image_path = root / "images" / split / f"{split}.jpg"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 24), color).save(image_path)
        records.append(
            {
                "sample_id": split,
                "split": split,
                "source_id": "one-provider",
                "source_group_id": "shared-video",
                "sha256": sha256_file(image_path),
                "phash": perceptual_hash(image_path),
                "human_review_status": "PENDING",
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    dataset = audit_dataset("TEST", root, manifest)
    report = build_report([("TEST", root, manifest)])

    assert dataset["governance_status"] == "NOT_GOVERNED"
    assert dataset["metadata"]["person_id"]["status"] == "UNKNOWN"
    assert dataset["subject_disjoint_status"] == "NOT_PROVABLE"
    assert dataset["leakage"]["source_group_id"]["status"] == "FAIL"
    assert dataset["resolution"]["distribution"] == {"32x24": 2}
    assert report["scientific_claims"]["subject_disjoint"] == "NOT_PROVABLE"
