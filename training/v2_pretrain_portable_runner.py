from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from training.common import sha256_file
from training.train_multimodel_v2 import train_suite
from training.v2_portable_runner import (
    _validate_dataset,
    _verify_hashes,
    runtime_config,
    select_profile,
    write_runtime_config,
)

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = Path("experiments/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL/config.yaml")
PROFILE_CONFIG = Path("experiments/MULTIMODEL_V2/profiles.yaml")
READINESS_REPORT = Path("reports/v2_pretrain_pending_approval_readiness.json")
EXPECTED_CLASSES = (
    "seatbelt_fastened",
    "seatbelt_unfastened",
    "uncertain_or_occluded",
)
COMPONENTS = {
    "phone_detector": {
        "dataset": Path("datasets/derived/v2_pretrain_pending_approval/phone_detector"),
        "classes": ["phone"],
        "audit": READINESS_REPORT,
    },
    "seatbelt_detector": {
        "dataset": Path("datasets/derived/v2_pretrain_pending_approval/seatbelt_detector"),
        "classes": ["occupant_upper_body"],
        "audit": READINESS_REPORT,
    },
}


def _validate_classifier(root: Path) -> dict:
    dataset = root / "datasets/derived/v2_pretrain_pending_approval/seatbelt_classifier"
    report = json.loads((dataset / "CLASSIFIER_DATASET_REPORT.json").read_text(encoding="utf-8"))
    if not report.get("ready_for_exploratory_training"):
        raise ValueError(
            "STOP: model-assisted pending-approval classifier is not exploratory-training ready"
        )
    if report.get("ready_for_governed_training"):
        raise ValueError(
            "STOP: model-assisted pending-approval classifier must not claim governed readiness"
        )
    manifest = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: defaultdict[str, set[str]] = defaultdict(set)
    manifest_counts = {split: Counter() for split in ("train", "val", "test")}
    for item in manifest:
        if (
            item.get("reviewer_type") != "AUTOMATED"
            or item.get("usage") != "MODEL_ASSISTED_PENDING_APPROVAL"
            or item.get("governance_eligible") is not False
        ):
            raise ValueError(f"STOP: invalid automated-review provenance: {item.get('sample_id')}")
        groups[str(item["effective_group_id"])].add(str(item["split"]))
        manifest_counts[item["split"]][item["class_name"]] += 1
    leaking = [group for group, splits in groups.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"STOP: classifier effective-group leakage: {len(leaking)} groups")

    disk_counts = {split: Counter() for split in ("train", "val", "test")}
    for split in disk_counts:
        for class_name in EXPECTED_CLASSES:
            class_dir = dataset / split / class_name
            images = [path for path in class_dir.iterdir() if path.is_file()]
            if not images:
                raise ValueError(f"STOP: empty classifier class: {split}/{class_name}")
            disk_counts[split][class_name] = len(images)
        if disk_counts[split] != manifest_counts[split]:
            raise ValueError(f"STOP: classifier manifest/file count mismatch: {split}")
    return {
        "classes": list(EXPECTED_CLASSES),
        "manifest_samples": len(manifest),
        "splits": {split: dict(counts) for split, counts in disk_counts.items()},
        "effective_groups": len(groups),
    }


def _validate_weights(root: Path, config: dict) -> dict:
    result = {}
    for component, payload in config["components"].items():
        relative = Path(payload["model"])
        weight = root / relative
        provenance_path = weight.with_suffix(".provenance.json")
        if not weight.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"STOP: missing offline weight/provenance: {relative}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        digest = sha256_file(weight)
        if (
            provenance.get("artifact") != weight.name
            or provenance.get("sha256") != digest
            or int(provenance.get("bytes", -1)) != weight.stat().st_size
        ):
            raise ValueError(f"STOP: offline base-weight provenance mismatch: {relative}")
        result[component] = {
            "path": relative.as_posix(),
            "bytes": weight.stat().st_size,
            "sha256": digest,
        }
    return result


def preflight(root: Path) -> dict:
    root = root.resolve()
    verified = _verify_hashes(root)
    readiness = json.loads((root / READINESS_REPORT).read_text(encoding="utf-8"))
    state = readiness.get("readiness", {})
    if readiness.get("status") != "PASS" or not state.get("proposal_detector_training_ready"):
        raise ValueError("STOP: pending-approval proposal-detector readiness audit is not PASS")
    if state.get("human_verified") or state.get("governed_training_ready"):
        raise ValueError("STOP: inconsistent automation-only governance claims in readiness report")
    config = yaml.safe_load((root / BASE_CONFIG).read_text(encoding="utf-8"))
    configured = set(config.get("components", {}))
    if not set(COMPONENTS).issubset(configured):
        raise ValueError("STOP: portable config is missing one or more proposal detectors")
    unexpected = configured - {*COMPONENTS, "seatbelt_classifier"}
    if unexpected:
        raise ValueError(f"STOP: unexpected portable components: {sorted(unexpected)}")
    if "seatbelt_classifier" in configured and not state.get("classifier_training_ready"):
        raise ValueError("STOP: blocked seatbelt classifier was included in the bundle")
    if config.get("governance_level") != "MODEL_ASSISTED_PENDING_APPROVAL":
        raise ValueError(
            "STOP: portable config governance level is not MODEL_ASSISTED_PENDING_APPROVAL"
        )
    profiles = yaml.safe_load((root / PROFILE_CONFIG).read_text(encoding="utf-8"))
    return {
        "verified_files": verified,
        "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
        "human_verified": False,
        "components": {
            **{
                name: _validate_dataset(root, name, specification)
                for name, specification in COMPONENTS.items()
            },
            **(
                {"seatbelt_classifier": _validate_classifier(root)}
                if "seatbelt_classifier" in configured
                else {}
            ),
        },
        "base_weights": _validate_weights(root, config),
        "profiles": sorted(profiles["profiles"]),
        "policy": "PRETRAIN_PENDING_APPROVAL_ONLY; FROZEN_TEST_AFTER_MODEL_LOCK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable model-assisted pending-approval V2 training runner"
    )
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument(
        "--working",
        type=Path,
        default=Path(os.environ.get("V2_WORKING", "runs/v2_pretrain_pending_approval")),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["auto", "rtx5060ti_8gb", "rtx5060ti_16gb"],
        default="auto",
    )
    parser.add_argument("--only", nargs="*", choices=sorted({*COMPONENTS, "seatbelt_classifier"}))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(os.environ["V2_BACKUP_DIR"]) if os.environ.get("V2_BACKUP_DIR") else None,
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    report = preflight(args.bundle)
    print(json.dumps({"preflight": report}, indent=2))
    if args.preflight_only:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA GPU unavailable; run this bundle on the training machine")
    bundle_root = args.bundle.resolve()
    working = args.working.resolve()
    profile_payload = yaml.safe_load((bundle_root / PROFILE_CONFIG).read_text(encoding="utf-8"))
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    profile = select_profile(profile_payload, args.profile, vram_gib)
    base = yaml.safe_load((bundle_root / BASE_CONFIG).read_text(encoding="utf-8"))
    runtime = runtime_config(
        base,
        profile_payload,
        profile,
        set(args.only or []),
        args.smoke,
        readiness_level="pretrain_pending",
    )
    config_path = write_runtime_config(working, runtime)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "vram_gib": round(vram_gib, 2),
                "profile": profile,
                "components": sorted(runtime["components"]),
                "smoke": args.smoke,
                "resume": args.resume,
                "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
                "backup_dir": str(args.backup_dir.resolve()) if args.backup_dir else None,
            },
            indent=2,
        )
    )
    os.chdir(bundle_root)
    result = train_suite(
        config_path,
        working,
        resume=args.resume,
        backup_dir=args.backup_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
