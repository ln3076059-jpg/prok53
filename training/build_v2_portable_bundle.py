from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from training.audit_v2_readiness import audit as audit_readiness
from training.audit_v2_readiness import write as write_readiness
from training.common import sha256_file

MAX_BUNDLE_BYTES = 1024**3
DATASETS = (
    Path("datasets/derived/phone_bootstrap_v2"),
    Path("datasets/derived/seatbelt_v2_balanced"),
)
AUDITS = (
    Path("reports/phone_bootstrap_v2/data_quality.json"),
    Path("reports/seatbelt_v2/audit.json"),
)
SUPPORT = (
    Path("START_V2_TRAINING.ps1"),
    Path("training/__init__.py"),
    Path("training/common.py"),
    Path("training/audit_v2_readiness.py"),
    Path("training/epoch_snapshots.py"),
    Path("training/train_multimodel_v2.py"),
    Path("training/v2_portable_runner.py"),
    Path("training/download_base_weights.py"),
    Path("requirements-kaggle.txt"),
    Path("requirements-windows-cu128.txt"),
    Path("experiments/MULTIMODEL_V2/profiles.yaml"),
    Path("models/base/yolo11s.pt"),
    Path("models/base/yolo11s.provenance.json"),
    Path("models/model_config_v2.yaml"),
    Path("datasets/derived/seatbelt_classifier_v2/CLASSIFIER_DATASET_REPORT.json"),
)
DETECTOR_COMPONENTS = ("phone_detector", "seatbelt_detector")
CLASSIFIER_REPORT = Path("datasets/derived/seatbelt_classifier_v2/CLASSIFIER_DATASET_REPORT.json")
READINESS_REPORT = Path("reports/v2_training_readiness.json")
READINESS_MARKDOWN = Path("reports/v2_training_readiness.md")


def _copy_relative(source: Path, root: Path) -> None:
    target = root / source
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.is_file():
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(source)


def build(config_path: Path, output: Path, max_bytes: int = MAX_BUNDLE_BYTES) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    components = config.get("components", {})
    missing = set(DETECTOR_COMPONENTS) - set(components)
    if missing:
        raise ValueError(f"missing detector components: {sorted(missing)}")
    portable_config = {
        **config,
        "components": {
            name: {**components[name], "model": "models/base/yolo11s.pt"}
            for name in DETECTOR_COMPONENTS
        },
    }
    classifier_report = json.loads(CLASSIFIER_REPORT.read_text(encoding="utf-8"))
    governance = audit_readiness()
    write_readiness(governance, READINESS_REPORT)
    readiness = {
        "schema_version": 1,
        "bundle_status": "TRAINABLE_PROPOSAL_DETECTORS_ONLY",
        "trainable_components": list(DETECTOR_COMPONENTS),
        "blocked_components": {
            "seatbelt_classifier": {
                "reason": (
                    "Every split requires human-reviewed FASTENED, UNFASTENED, and "
                    "UNCERTAIN_OR_OCCLUDED crops before governed training."
                ),
                "ready_for_training": bool(classifier_report.get("ready_for_training")),
                "counts": classifier_report.get("counts", {}),
            }
        },
        "data_policy": (
            "Detector labels remain PENDING/PROPOSAL_MODEL_ONLY. Training produces proposal "
            "weights, not semantically approved production ground truth."
        ),
        "one_command_windows": ".\\START_V2_TRAINING.ps1",
        "resume_policy": "SAFE_AUTO_RESUME_WHEN_PLAN_MATCHES_AND_LAST_PT_EXISTS",
        "gpu_smoke_policy": "ONE_EPOCH_PER_COMPONENT_BEFORE_FULL_TRAIN_BY_DEFAULT",
        "external_backup_supported": True,
        "default_training_gate": "GOVERNED_READINESS_REQUIRED",
        "governance_status": governance["status"],
        "governance_blocker_codes": governance["blocker_codes"],
        "governance_report": str(READINESS_REPORT).replace("\\", "/"),
    }
    if readiness["blocked_components"]["seatbelt_classifier"]["ready_for_training"]:
        raise ValueError(
            "classifier is ready but the detector-only bundle definition has not been promoted; "
            "refuse to silently omit a trainable component"
        )

    for audit_path in AUDITS:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise ValueError(f"audit must pass before bundling: {audit_path}")

    with tempfile.TemporaryDirectory(prefix="v2-portable-") as temp_dir:
        root = Path(temp_dir) / "MULTIMODEL_V2_PORTABLE"
        root.mkdir()
        for source in (
            *DATASETS,
            *AUDITS,
            *SUPPORT,
            READINESS_REPORT,
            READINESS_MARKDOWN,
        ):
            _copy_relative(source, root)
        target_config = root / "experiments/MULTIMODEL_V2/config.yaml"
        target_config.parent.mkdir(parents=True, exist_ok=True)
        target_config.write_text(yaml.safe_dump(portable_config, sort_keys=False), encoding="utf-8")
        (root / "TRAINING_READINESS.json").write_text(
            json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8"
        )
        (root / "README_TRAIN.txt").write_text(
            "MULTIMODEL V2 detector-only portable bundle\n"
            "Governed train (safe default): .\\START_V2_TRAINING.ps1\n"
            "Current pending data will be blocked until review and bundle rebuild.\n"
            "Explicit proposal experiment only: add -AllowProposalTraining.\n"
            "First machine setup: add -InstallDependencies.\n"
            "Preflight only: .\\START_V2_TRAINING.ps1 -Mode preflight\n"
            "Smoke only: .\\START_V2_TRAINING.ps1 -Mode smoke -Working runs/smoke\n"
            "Resume: repeat the exact same train command; safe auto-resume is the default.\n"
            "GPU smoke: one epoch/component runs before full training unless -SkipSmoke is used.\n"
            "Second-disk backup: add -BackupDirectory X:\\v2-backup.\n"
            "PowerShell isolation: add -Only phone_detector or -Only seatbelt_detector.\n"
            "Profile auto selects the governed RTX 5060 Ti 8 GB or 16 GB settings.\n"
            "The runner uses validation only. Keep test frozen until the model is locked.\n",
            encoding="utf-8",
        )
        expected = {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        (root / "V2_PORTABLE_EXPECTED_HASHES.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        archive = Path(shutil.make_archive(str(output.with_suffix("")), "zip", root))

    size = archive.stat().st_size
    if size > max_bytes:
        archive.unlink()
        raise ValueError(f"portable bundle exceeded byte limit: {size} > {max_bytes}")
    digest = sha256_file(archive)
    checksum = archive.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "bundle": str(archive.resolve()),
        "checksum": str(checksum.resolve()),
        "bytes": size,
        "size_mib": round(size / 1024**2, 2),
        "limit_bytes": max_bytes,
        "under_limit": True,
        "sha256": digest,
        "components": list(DETECTOR_COMPONENTS),
        "classifier_included": False,
        "readiness": readiness,
        "test_policy": "FROZEN_UNTIL_MODEL_LOCK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the under-1-GiB portable V2 detector bundle"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/MULTIMODEL_V2/config.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("kaggle/MULTIMODEL_V2_PORTABLE.zip"))
    parser.add_argument("--max-bytes", type=int, default=MAX_BUNDLE_BYTES)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output, args.max_bytes), indent=2))


if __name__ == "__main__":
    main()
