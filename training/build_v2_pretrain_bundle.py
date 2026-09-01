from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from training.audit_pretrain_pending_v2 import audit
from training.common import sha256_file

MAX_BUNDLE_BYTES = 1024**3
CONFIG = Path("experiments/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL/config.yaml")
READINESS = Path("reports/v2_pretrain_pending_approval_readiness.json")
DATASET_ROOT = Path("datasets/derived/v2_pretrain_pending_approval")
DETECTOR_COMPONENTS = ("phone_detector", "seatbelt_detector")
BASE_SOURCES = (
    DATASET_ROOT / "phone_detector",
    DATASET_ROOT / "seatbelt_detector",
    DATASET_ROOT / "PRETRAIN_PENDING_APPROVAL_DATASET_REPORT.json",
    Path(
        "datasets/manifests/pretrain_pending_approval/"
        "v2_seatbelt_uncertain_visual_decisions.json"
    ),
    CONFIG,
    Path("experiments/MULTIMODEL_V2/profiles.yaml"),
    READINESS,
    Path("models/base/yolo11s.pt"),
    Path("models/base/yolo11s.provenance.json"),
    Path("training/__init__.py"),
    Path("training/common.py"),
    Path("training/train_multimodel_v2.py"),
    Path("training/v2_portable_runner.py"),
    Path("training/v2_pretrain_portable_runner.py"),
    Path("START_V2_PRETRAIN_REVIEW_TRAINING.ps1"),
    Path("requirements-kaggle.txt"),
    Path("requirements-windows-cu128.txt"),
)


def _copy_relative(source: Path, root: Path) -> None:
    target = root / source
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.is_file():
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(source)


def build(output: Path, max_bytes: int = MAX_BUNDLE_BYTES) -> dict:
    readiness = audit()
    if readiness["status"] != "PASS":
        raise ValueError(
            f"model-assisted pending-approval readiness audit failed: {readiness['errors']}"
        )
    state = readiness["readiness"]
    if not state["proposal_detector_training_ready"]:
        raise ValueError(
            "model-assisted pending-approval detector training is blocked: "
            f"{readiness.get('training_blockers', [])}"
        )
    recorded = json.loads(READINESS.read_text(encoding="utf-8"))
    if recorded != readiness:
        raise ValueError("readiness report is stale; run training.audit_pretrain_pending_v2")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trainable = list(DETECTOR_COMPONENTS)
    classifier_included = bool(state["classifier_training_ready"])
    sources = list(BASE_SOURCES)
    if classifier_included:
        trainable.append("seatbelt_classifier")
        sources.extend(
            [
                DATASET_ROOT / "seatbelt_classifier",
                Path("models/base/yolo11s-cls.pt"),
                Path("models/base/yolo11s-cls.provenance.json"),
            ]
        )
    portable_config = {
        **config,
        "components": {name: config["components"][name] for name in trainable},
    }
    classifier_note = (
        "The classifier is included because its uncertainty evidence minimum is met.\n"
        if classifier_included
        else "The classifier is excluded until its uncertainty evidence minimum is met.\n"
    )

    with tempfile.TemporaryDirectory(prefix="v2-pretrain-pending-") as temp_dir:
        root = Path(temp_dir) / "MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE"
        root.mkdir()
        for source in sources:
            _copy_relative(source, root)
        target_config = root / CONFIG
        target_config.write_text(yaml.safe_dump(portable_config, sort_keys=False), encoding="utf-8")
        component_text = ", ".join(trainable)
        (root / "README_TRAIN.txt").write_text(
            "MULTIMODEL V2 — model-assisted pending-approval exploratory training bundle\n"
            "Governance: MODEL_ASSISTED_PENDING_APPROVAL; this is not "
            "HUMAN_APPROVED or production-ready.\n"
            f"Trainable components in this build: {component_text}.\n"
            f"{classifier_note}"
            "1) Preflight: .\\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -Mode preflight\n"
            "2) Train: .\\START_V2_PRETRAIN_REVIEW_TRAINING.ps1\n"
            "The train command runs a one-epoch smoke suite first.\n"
            "Resume: repeat the same command and working directory.\n"
            "Second-disk backup: add -BackupDirectory X:\\v2-pretrain-backup\n"
            "Install once if needed: add -InstallDependencies\n"
            "Component isolation: add -Only phone_detector, seatbelt_detector, "
            "or seatbelt_classifier.\n"
            "No Internet is required for bundled YOLO base weights.\n"
            "Training uses validation only; keep test frozen until model lock.\n",
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
        "sha256": digest,
        "components": trainable,
        "classifier_included": classifier_included,
        "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
        "human_verified": False,
        "proposal_detector_training_ready": True,
        "complete_suite_training_ready": bool(state["exploratory_training_ready"]),
        "governed_training_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build portable model-assisted pending-approval V2 bundle"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip"),
    )
    parser.add_argument("--max-bytes", type=int, default=MAX_BUNDLE_BYTES)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.max_bytes), indent=2))


if __name__ == "__main__":
    main()
