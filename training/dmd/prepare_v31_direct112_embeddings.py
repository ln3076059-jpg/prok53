"""Create versioned direct-112 RGB caches and frozen ResNet-18 embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common import sha256_file, stable_json_hash
from training.dmd.research_permissions import require_operation
from training.dmd.run_v5_experiments import DEVELOPMENT_SUBJECTS, discover_external_dmd_root
from training.dmd.v5_resnet_gru import extract_resnet18_embeddings
from training.dmd.v5_video_action import extract_rgb_cache


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def run(manifest_path: Path) -> dict[str, Any]:
    require_operation("local_dmd_training")
    root = discover_external_dmd_root()
    derivatives = root / "derivatives"
    rgb_dir = derivatives / "v31_rgb_cache_direct112"
    embedding_dir = derivatives / "v31_resnet18_embeddings_direct112"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETED":
            raise FileExistsError("completed direct112 manifest is immutable")
        if manifest.get("preprocessing_id") != "DIRECT_SOURCE_BODY_ROI_112_FLOAT64_V31":
            raise ValueError("cannot resume a different preprocessing contract")
        subjects = manifest["subjects"]
    else:
        subjects: list[dict[str, Any]] = []
        manifest = {
            "schema_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "RUNNING",
            "preprocessing_id": "DIRECT_SOURCE_BODY_ROI_112_FLOAT64_V31",
            "subjects": subjects,
            "governance": {
                "execution_location": "LOCAL_ONLY",
                "dmd_media_uploaded": False,
                "consumed_holdouts_accessed": [],
            },
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed_subjects = {row["subject"] for row in subjects}
    for subject in DEVELOPMENT_SUBJECTS:
        subject_dir = root / "subjects" / subject
        video_path = _find_one(subject_dir, "*_rgb_body.mp4")
        annotation_path = _find_one(subject_dir, "*_rgb_ann_distraction.json")
        feature_path = derivatives / f"v3_features_{subject}.json"
        rgb_path = rgb_dir / f"v31_rgb_{subject}_112.npz"
        embedding_path = embedding_dir / f"v31_resnet18_{subject}.npz"
        if subject in completed_subjects:
            if not rgb_path.is_file() or not embedding_path.is_file():
                raise FileNotFoundError(
                    f"resume manifest references missing artifact for {subject}"
                )
            continue
        if rgb_path.exists() or embedding_path.exists():
            raise FileExistsError(
                f"immutable direct112 artifact already exists for {subject}"
            )
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        rgb_receipt = extract_rgb_cache(
            subject,
            video_path,
            feature,
            annotation_path,
            rgb_path,
            image_size=112,
        )
        embedding_receipt = extract_resnet18_embeddings(
            subject,
            rgb_path,
            embedding_path,
            batch_size=64,
        )
        subjects.append(
            {
                "subject": subject,
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
                "rgb": rgb_receipt,
                "embedding": embedding_receipt,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    manifest["status"] = "COMPLETED"
    manifest["subject_set_sha256"] = stable_json_hash(subjects)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reports/v31_ai85_plan_execution/direct112_embedding_manifest.json"
        ),
    )
    args = parser.parse_args()
    result = run(args.manifest)
    print(
        json.dumps(
            {
                "status": result["status"],
                "preprocessing_id": result["preprocessing_id"],
                "subjects": [row["subject"] for row in result["subjects"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
