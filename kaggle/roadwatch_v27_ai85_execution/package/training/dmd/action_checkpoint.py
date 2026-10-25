"""Fail-closed contracts for research action checkpoints; no fold substitution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

COARSE_CLASSES = ["SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY"]
FEATURE_CONTRACT = {
    "backbone": "torchvision_resnet18_imagenet1k_v1",
    "embedding_dim": 512,
    "crop_roi_normalized": [0.0, 0.10, 0.65, 0.95],
    "resize": "opencv_area_64_then_torch_bilinear_112_align_corners_false",
    "color": "RGB",
    "normalization": "IMAGENET_MEAN_STD",
    "embedding_storage": "float16_then_float32",
    "sequence_length": 9,
    "padding": "NONE_FULL_WINDOWS_ONLY",
    "label_position": "CENTER",
}


class CheckpointSafetyError(ValueError):
    """A checkpoint cannot substantiate the requested inference contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_manifest(
    checkpoint: Path,
    manifest_path: Path,
    subject: str,
    architecture: str,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise CheckpointSafetyError("MISSING_COMPATIBLE_LOSO_CHECKPOINT")
    if not manifest_path.is_file():
        raise CheckpointSafetyError("CHECKPOINT_PROVENANCE_NOT_PROVABLE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "training_subject_ids",
        "validation_subject_ids",
        "intended_heldout_subject",
        "architecture",
        "feature_preprocessing",
        "class_mapping",
        "sha256",
        "training_dataset_provenance",
        "backbone_sha256",
        "sample_period_seconds",
    }
    if not required <= manifest.keys():
        raise CheckpointSafetyError("CHECKPOINT_PROVENANCE_NOT_PROVABLE")
    train = manifest["training_subject_ids"]
    validation = manifest["validation_subject_ids"]
    if not isinstance(train, list) or not train or not isinstance(validation, list):
        raise CheckpointSafetyError("CHECKPOINT_PROVENANCE_NOT_PROVABLE")
    if subject != manifest["intended_heldout_subject"]:
        raise CheckpointSafetyError("WRONG_LOSO_FOLD")
    if subject in train or subject in validation or set(train) & set(validation):
        raise CheckpointSafetyError("TRAIN_VALIDATION_HELDOUT_OVERLAP")
    if set(train + validation) - {"gB-9", "gC-14", "gZ-36"}:
        raise CheckpointSafetyError("PROTECTED_OR_UNAUTHORIZED_FITTING_SUBJECT")
    if manifest["architecture"] != architecture:
        raise CheckpointSafetyError("INCOMPATIBLE_CHECKPOINT_ARCHITECTURE")
    if manifest["feature_preprocessing"] != FEATURE_CONTRACT:
        raise CheckpointSafetyError("FEATURE_PREPROCESSING_MISMATCH")
    if manifest["class_mapping"] != COARSE_CLASSES:
        raise CheckpointSafetyError("ACTION_CLASS_MAPPING_MISMATCH")
    if sha256_file(checkpoint) != manifest["sha256"]:
        raise CheckpointSafetyError("CHECKPOINT_SHA_MISMATCH")
    evidence = manifest["training_dataset_provenance"]
    if not isinstance(evidence, list) or not evidence:
        raise CheckpointSafetyError("CHECKPOINT_PROVENANCE_NOT_PROVABLE")
    for item in evidence:
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise CheckpointSafetyError("TRAINING_PROVENANCE_SHA_MISMATCH")
    if not isinstance(manifest["sample_period_seconds"], int | float) or not (
        0 < manifest["sample_period_seconds"] < 60
    ):
        raise CheckpointSafetyError("INVALID_SAMPLE_PERIOD")
    return manifest
