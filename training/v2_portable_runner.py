from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import yaml

from training.common import sha256_file
from training.train_multimodel_v2 import train_suite

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_HASHES = "V2_PORTABLE_EXPECTED_HASHES.json"
PROFILE_CONFIG = Path("experiments/MULTIMODEL_V2/profiles.yaml")
BASE_CONFIG = Path("experiments/MULTIMODEL_V2/config.yaml")
BASE_WEIGHT_PROVENANCE = Path("models/base/yolo11s.provenance.json")
READINESS_REPORT = Path("TRAINING_READINESS.json")
COMPONENTS = {
    "phone_detector": {
        "dataset": Path("datasets/derived/phone_bootstrap_v2"),
        "classes": ["phone"],
        "audit": Path("reports/phone_bootstrap_v2/data_quality.json"),
    },
    "seatbelt_detector": {
        "dataset": Path("datasets/derived/seatbelt_v2_balanced"),
        "classes": ["occupant_upper_body"],
        "audit": Path("reports/seatbelt_v2/audit.json"),
    },
}


def _names(payload: dict) -> list[str]:
    names = payload.get("names")
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=int)]
    raise ValueError("STOP: data.yaml names must be a list or integer-keyed mapping")


def _verify_hashes(root: Path) -> int:
    manifest_path = root / EXPECTED_HASHES
    if not manifest_path.is_file():
        raise FileNotFoundError(f"STOP: missing {EXPECTED_HASHES}")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"STOP: bundle hash mismatch: {relative}")
    return len(expected)


def _validate_yolo_label(path: Path, class_count: int) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"STOP: malformed label {path}:{line_number}")
        class_id = int(parts[0])
        x, y, width, height = [float(value) for value in parts[1:]]
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"STOP: non-finite box {path}:{line_number}")
        if not 0 <= class_id < class_count:
            raise ValueError(f"STOP: class id out of range {path}:{line_number}")
        if width <= 0 or height <= 0:
            raise ValueError(f"STOP: non-positive box {path}:{line_number}")
        if x - width / 2 < -1e-9 or x + width / 2 > 1 + 1e-9:
            raise ValueError(f"STOP: horizontal box outside image {path}:{line_number}")
        if y - height / 2 < -1e-9 or y + height / 2 > 1 + 1e-9:
            raise ValueError(f"STOP: vertical box outside image {path}:{line_number}")
        boxes += 1
    return boxes


def _validate_dataset(root: Path, component: str, specification: dict) -> dict:
    dataset = root / specification["dataset"]
    payload = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = _names(payload)
    if names != specification["classes"]:
        raise ValueError(
            f"STOP: {component} class order is {names}, expected {specification['classes']}"
        )
    if "nc" in payload and int(payload["nc"]) != len(names):
        raise ValueError(f"STOP: {component} nc does not match names")

    audit = json.loads((root / specification["audit"]).read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError(f"STOP: {component} audit is not PASS")

    manifest = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overlaps: dict[str, list[str]] = {}
    for key in ("sha256", "source_group_id", "effective_group_id", "near_cluster_id"):
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            value = item.get(key)
            if value:
                seen[str(value)].add(str(item["split"]))
        overlaps[key] = sorted(value for value, splits in seen.items() if len(splits) > 1)
    if any(overlaps.values()):
        counts = {key: len(values) for key, values in overlaps.items()}
        raise ValueError(f"STOP: {component} split isolation failure: {counts}")

    splits: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        if not images:
            raise ValueError(f"STOP: empty {component}/{split} split")
        boxes = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"STOP: missing label for {image}")
            boxes += _validate_yolo_label(label, len(names))
        splits[split] = {"images": len(images), "boxes": boxes}
    return {"classes": names, "manifest_samples": len(manifest), "splits": splits}


def _validate_base_weights(root: Path, config: dict) -> dict[str, dict]:
    provenance = json.loads((root / BASE_WEIGHT_PROVENANCE).read_text(encoding="utf-8"))
    report = {}
    for component, payload in config.get("components", {}).items():
        model = Path(str(payload.get("model", "")))
        path = (root / model).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"STOP: model path escapes bundle: {model}") from error
        if not path.is_file():
            raise FileNotFoundError(f"STOP: bundled base weight is missing: {model}")
        digest = sha256_file(path)
        if path.name != provenance.get("artifact"):
            raise ValueError(f"STOP: base weight provenance name mismatch: {model}")
        if path.stat().st_size != int(provenance.get("bytes", -1)):
            raise ValueError(f"STOP: base weight provenance size mismatch: {model}")
        if digest != provenance.get("sha256"):
            raise ValueError(f"STOP: base weight provenance SHA mismatch: {model}")
        report[component] = {
            "path": str(model).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "source_url": provenance["source_url"],
        }
    return report


def select_profile(profile_config: dict, requested: str, vram_gib: float) -> str:
    profiles = profile_config.get("profiles", {})
    if not profiles:
        raise ValueError("STOP: no portable GPU profiles are defined")
    if requested == "auto":
        split = float(profile_config["auto_profile_vram_split_gib"])
        selected = "rtx5060ti_8gb" if vram_gib < split else "rtx5060ti_16gb"
    else:
        selected = requested
    if selected not in profiles:
        raise ValueError(f"STOP: unknown GPU profile: {selected}")
    minimum = float(profiles[selected]["min_vram_gib"])
    if vram_gib + 0.05 < minimum:
        raise ValueError(
            f"STOP: {selected} needs at least {minimum:.1f} GiB VRAM; detected {vram_gib:.2f}"
        )
    return selected


def runtime_config(
    base: dict,
    profile_config: dict,
    profile: str,
    selected_components: set[str],
    smoke: bool,
    readiness_level: str = "governed",
) -> dict:
    payload = copy.deepcopy(base)
    components = payload.get("components", {})
    unknown = selected_components - set(components)
    if unknown:
        raise ValueError(f"STOP: unknown components: {sorted(unknown)}")
    if selected_components:
        components = {
            name: item for name, item in components.items() if name in selected_components
        }
    overrides = profile_config["profiles"][profile].get("components", {})
    for name, item in components.items():
        item.update(overrides.get(name, {}))
        if smoke:
            item.update({"epochs": 1, "patience": 1, "save_period": 1, "cache": False})
            if item.get("task") == "detect":
                item["multi_scale"] = False
    selection = "all" if not selected_components else "_".join(sorted(selected_components))
    mode = "SMOKE" if smoke else "TRAIN"
    payload["experiment_id"] = (
        f"MULTIMODEL_V2_{profile}_{selection}_{readiness_level}_{mode}".upper()
    )
    payload["components"] = components
    payload["runtime_policy"] = {
        "profile": profile,
        "selection": sorted(components),
        "smoke": smoke,
        "test_split_used": False,
        "readiness_level": readiness_level,
    }
    return payload


def write_runtime_config(working: Path, payload: dict) -> Path:
    working.mkdir(parents=True, exist_ok=True)
    target = working / "runtime_config.yaml"
    text = yaml.safe_dump(payload, sort_keys=False)
    if target.is_file() and target.read_text(encoding="utf-8") != text:
        raise ValueError("STOP: working directory contains a different runtime config")
    target.write_text(text, encoding="utf-8")
    return target


def preflight(root: Path) -> dict:
    root = root.resolve()
    verified = _verify_hashes(root)
    config = yaml.safe_load((root / BASE_CONFIG).read_text(encoding="utf-8"))
    if set(config.get("components", {})) != set(COMPONENTS):
        raise ValueError("STOP: portable config must contain detector components only")
    profiles = yaml.safe_load((root / PROFILE_CONFIG).read_text(encoding="utf-8"))
    readiness = json.loads((root / READINESS_REPORT).read_text(encoding="utf-8"))
    if set(readiness.get("trainable_components", [])) != set(config.get("components", {})):
        raise ValueError("STOP: readiness component list does not match portable config")
    governance = readiness.get("governance_status", {})
    if not governance.get("proposal_detector_training_ready"):
        raise ValueError("STOP: proposal detector readiness gate is not satisfied")
    return {
        "verified_files": verified,
        "components": {
            name: _validate_dataset(root, name, specification)
            for name, specification in COMPONENTS.items()
        },
        "base_weights": _validate_base_weights(root, config),
        "profiles": profiles,
        "readiness": readiness,
        "policy": "VALIDATION_ONLY_DURING_TRAINING; FROZEN_TEST_AFTER_MODEL_LOCK",
    }


def enforce_readiness(report: dict, level: str) -> None:
    readiness = report["readiness"]
    governance = readiness.get("governance_status", {})
    key = "proposal_detector_training_ready" if level == "proposal" else "governed_training_ready"
    if governance.get(key):
        return
    blockers = readiness.get("governance_blocker_codes", [])
    raise ValueError(
        f"STOP: {level} readiness is false; resolve and rebuild the bundle. "
        f"Blockers: {', '.join(blockers)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed portable V2 detector runner")
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument(
        "--working",
        type=Path,
        default=Path(os.environ.get("V2_WORKING", "runs/v2_portable")),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["auto", "rtx5060ti_8gb", "rtx5060ti_16gb"],
        default="auto",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(COMPONENTS),
        help="Train one or both detector components in an isolated experiment",
    )
    parser.add_argument("--smoke", action="store_true", help="Run one epoch per component")
    parser.add_argument(
        "--readiness-level",
        choices=["governed", "proposal"],
        default="governed",
        help="Governed is the safe default; proposal requires an explicit opt-in",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(os.environ["V2_BACKUP_DIR"]) if os.environ.get("V2_BACKUP_DIR") else None,
        help="Optional second disk/cloud-synced directory for per-epoch resume checkpoints",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Explicitly enable safe resume (already the default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Refuse to continue an existing incomplete run",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    report = preflight(args.bundle)
    print(json.dumps({"preflight": report}, indent=2))
    enforce_readiness(report, args.readiness_level)
    if args.preflight_only:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA GPU is unavailable; move this bundle to the GPU machine")
    bundle_root = args.bundle.resolve()
    working = args.working.resolve()
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    profile_config = yaml.safe_load((bundle_root / PROFILE_CONFIG).read_text(encoding="utf-8"))
    profile = select_profile(profile_config, args.profile, vram_gib)
    base = yaml.safe_load((bundle_root / BASE_CONFIG).read_text(encoding="utf-8"))
    runtime = runtime_config(
        base,
        profile_config,
        profile,
        set(args.only or []),
        args.smoke,
        readiness_level=args.readiness_level,
    )
    config_path = write_runtime_config(working, runtime)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "vram_gib": round(vram_gib, 2),
                "profile": profile,
                "smoke": args.smoke,
                "components": sorted(runtime["components"]),
                "resume": args.resume,
                "readiness_level": args.readiness_level,
                "backup_dir": str(args.backup_dir.resolve()) if args.backup_dir else None,
                "runtime_config": str(config_path),
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
