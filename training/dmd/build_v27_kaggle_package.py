"""Build a private, code-only V27 Kaggle package with no DMD media or labels."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from training.dmd.research_permissions import require_operation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "kaggle" / "roadwatch_v27_ai85_execution"
DATASET_ID = "longngoc11/roadwatch-v27-ai85-execution"

FILES = (
    "backend/ai/association.py",
    "backend/ai/auxiliary.py",
    "backend/ai/cabin.py",
    "backend/ai/detector.py",
    "backend/ai/events.py",
    "backend/ai/fusion.py",
    "models/temporal_config_v2.json",
    "reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json",
    "reports/v27_ai85_execution/EXECUTION_REPORT.md",
    "reports/v27_ai85_execution/FINAL_AI_LEAD_DECISION.md",
    "reports/v27_ai85_execution/RAW_D5_REPORT.md",
    "reports/v27_ai85_execution/component_metrics.json",
    "reports/v27_ai85_execution/kaggle_phone_component_receipt.json",
    "reports/v27_ai85_execution/kaggle_phone_component_v2/v27_phone_component_eval.json",
    "reports/v27_ai85_execution/metrics_comparison.json",
    "reports/v27_ai85_execution/raw_d5_full_none/aggregate_metrics.json",
    "reports/v27_ai85_execution/runtime_manifest.json",
    "reports/v27_ai85_execution/test_results.json",
    "kaggle/v27_phone_component_eval/kernel-metadata.json",
    "kaggle/v27_phone_component_eval/run_phone_component_eval.py",
    "requirements-kaggle.txt",
    "training/common.py",
    "training/dmd/action_adapter.py",
    "training/dmd/action_checkpoint.py",
    "training/dmd/action_model.py",
    "training/dmd/adapter.py",
    "training/dmd/audit_v27_execution.py",
    "training/dmd/audit_v27_assoc_bypass.py",
    "training/dmd/audit_v27_handheld_upper_bound.py",
    "training/dmd/audit_v27_phone_component_kaggle.py",
    "training/dmd/audit_v27_raw_d5.py",
    "training/dmd/behavior_roi.py",
    "training/dmd/driver_phone_association.py",
    "training/dmd/holdout_guard.py",
    "training/dmd/research_permissions.py",
    "training/dmd/run_autonomous_v21.py",
    "training/dmd/run_verified_v24.py",
    "training/dmd/temporal_eval.py",
    "training/dmd/v5_resnet_gru.py",
    "training/dmd/v5_video_action.py",
    "training/dmd/v6_action_hierarchy.py",
    "training/dmd/v7_action_models.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    require_operation("upload")
    package = OUTPUT / "package"
    output_resolved = OUTPUT.resolve()
    package_resolved = package.resolve()
    if not package_resolved.is_relative_to(output_resolved):
        raise ValueError("UNSAFE_KAGGLE_PACKAGE_PATH")
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append(
            {
                "path": f"package/{relative}",
                "bytes": target.stat().st_size,
                "sha256": digest(target),
            }
        )
    copied_files = {
        path.relative_to(OUTPUT).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    }
    manifested_files = {entry["path"] for entry in entries}
    if copied_files != manifested_files:
        raise ValueError("KAGGLE_PACKAGE_CONTAINS_UNMANIFESTED_FILES")

    metadata = {
        "title": "Roadwatch V27 AI85 Execution",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": (
            "Private code-only V27 correctness package; no raw DMD media, labels, "
            "embeddings, checkpoints, credentials, or raw predictions."
        ),
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "file_count": len(entries),
        "files": entries,
        "exclusions": {
            "credentials": True,
            "raw_media": True,
            "annotations": True,
            "embeddings": True,
            "model_checkpoints": True,
            "raw_predictions": True,
        },
    }
    (OUTPUT / "package-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# Roadwatch V27 AI85 execution package

Private code-only package for the V27 candidate-identity, fail-fast model-mode,
centered-score alignment, shared-device fixes, and Kaggle phone-component audit.

No DMD video, frame, annotation, embedding, action checkpoint, raw prediction,
or credential is included. The cached-development reference is TP/FP/FN = 12/13/8,
precision = 0.48, recall = 0.60, and F1 = 0.533333.

This package supports code verification and GPU preflight only. It cannot produce
raw DMD event metrics unless an independently authorized input dataset is attached.
The included phone-component result is provisional: its bootstrap labels are model
proposals, not governed final ground truth, and it is not an event-level metric.
The included aggregate raw-development result is 14/3/6 TP/FP/FN with precision
0.8235, recall 0.7000, and F1 0.7568; the >0.85 gate remains unmet.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
