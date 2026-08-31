from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image

TRAIN_KEYS = {
    "task", "imgsz", "epochs", "patience", "batch", "device", "workers", "cache", "amp",
    "optimizer", "cos_lr", "seed", "deterministic", "save", "save_period", "degrees",
    "translate", "scale", "hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "mixup",
    "classes", "close_mosaic", "multi_scale", "warmup_epochs", "weight_decay",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_bundle(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = list(Path("/kaggle/input").rglob("PHONE_BOOTSTRAP_EXPECTED_HASHES.json"))
    if len(candidates) != 1:
        raise ValueError(f"STOP: expected one phone bootstrap bundle, found {len(candidates)}")
    return candidates[0].parent


def verify_hashes(root: Path) -> int:
    expected = json.loads((root / "PHONE_BOOTSTRAP_EXPECTED_HASHES.json").read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"STOP: bundle hash mismatch: {relative}")
    return len(expected)


def read_names(data: dict) -> list[str]:
    names = data.get("names", {})
    return list(names) if isinstance(names, list) else [names[index] for index in sorted(names)]


def validate_label(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"STOP: invalid one-class label {path}:{line_number}")
        x, y, width, height = (float(value) for value in parts[1:])
        if width <= 0 or height <= 0 or x - width / 2 < -1e-9 or x + width / 2 > 1 + 1e-9:
            raise ValueError(f"STOP: invalid horizontal box {path}:{line_number}")
        if y - height / 2 < -1e-9 or y + height / 2 > 1 + 1e-9:
            raise ValueError(f"STOP: invalid vertical box {path}:{line_number}")
        count += 1
    return count


def preflight(root: Path) -> dict:
    verified = verify_hashes(root)
    dataset = root / "dataset"
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if read_names(data) != ["phone"]:
        raise ValueError("STOP: bootstrap class order must be exactly [phone]")
    bootstrap = json.loads((dataset / "BOOTSTRAP_MANIFEST.json").read_text(encoding="utf-8"))
    if bootstrap.get("status") != "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA":
        raise ValueError("STOP: dataset is not marked proposal-model-only")
    if any(bootstrap.get("group_overlap", {}).values()):
        raise ValueError("STOP: source groups overlap splits")
    audit = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("near_duplicate_pairs") or audit.get("exact_duplicate_groups"):
        raise ValueError("STOP: duplicate/isolation audit did not pass")

    manifest = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overlaps = {}
    for key in ("source_group_id", "effective_group_id", "near_cluster_id", "sha256"):
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for item in manifest:
            if item.get(key):
                seen[str(item[key])].add(item["split"])
        overlaps[key] = [value for value, splits in seen.items() if len(splits) > 1]
    if any(overlaps.values()):
        raise ValueError(f"STOP: manifest isolation failure: { {key: len(value) for key, value in overlaps.items()} }")
    if any(item.get("usage") != "PROPOSAL_MODEL_ONLY" for item in manifest):
        raise ValueError("STOP: bootstrap usage policy mismatch")

    splits = {}
    for split in ("train", "val", "test"):
        images = sorted(path for path in (dataset / "images" / split).glob("*") if path.is_file())
        if not images:
            raise ValueError(f"STOP: empty split: {split}")
        boxes = 0
        for image_path in images:
            with Image.open(image_path) as image:
                image.verify()
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"STOP: missing label: {label_path.relative_to(root)}")
            boxes += validate_label(label_path)
        if boxes == 0:
            raise ValueError(f"STOP: split has no phone boxes: {split}")
        splits[split] = {"images": len(images), "phone_boxes": boxes}
    return {"verified_files": verified, "splits": splits, "manifest_samples": len(manifest)}


def environment() -> dict:
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("STOP: Kaggle GPU is unavailable")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__,
        "disk_free_bytes": shutil.disk_usage("/kaggle/working").free,
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout.splitlines(),
    }


def runtime_data_yaml(root: Path, working: Path) -> Path:
    data = yaml.safe_load((root / "dataset" / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str((root / "dataset").resolve())
    target = working / "phone_bootstrap_data.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def train(root: Path, working: Path) -> tuple[Path, dict]:
    from ultralytics import YOLO

    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    unknown = set(config) - TRAIN_KEYS - {"experiment_id", "model"}
    if unknown:
        raise ValueError(f"STOP: unapproved training keys: {sorted(unknown)}")
    model_name = config.pop("model")
    config.pop("experiment_id", None)
    config = {key: value for key, value in config.items() if key in TRAIN_KEYS}
    data_yaml = runtime_data_yaml(root, working)
    config.update(
        data=str(data_yaml),
        project=str(working),
        name="PHONE_BOOTSTRAP_001",
        exist_ok=False,
    )
    YOLO(model_name).train(**config)
    run = working / "PHONE_BOOTSTRAP_001"
    best = run / "weights" / "best.pt"
    metrics = YOLO(best).val(
        data=str(data_yaml), split="test", imgsz=int(config["imgsz"]), device=config["device"], verbose=False
    )
    test_metrics = {key: float(value) for key, value in metrics.results_dict.items()}
    (run / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8")
    return run, test_metrics


def package(run: Path, root: Path, preflight_report: dict, env: dict, test_metrics: dict, working: Path) -> Path:
    for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "test_metrics.json"):
        if not (run / relative).is_file():
            raise FileNotFoundError(f"missing training artifact: {relative}")
    metadata = {
        "experiment_id": "PHONE_BOOTSTRAP_001",
        "usage": "PROPOSAL_MODEL_ONLY",
        "best_weights_sha256": sha256_file(run / "weights" / "best.pt"),
        "preflight": preflight_report,
        "test_metrics": test_metrics,
        "environment": env,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return Path(shutil.make_archive(str(working / "PHONE_BOOTSTRAP_001_RECOVERY"), "zip", run))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed Kaggle phone proposal bootstrap runner")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--working", type=Path, default=Path(os.environ.get("PHONE_BOOTSTRAP_WORKING", "/kaggle/working"))
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = locate_bundle(args.bundle)
    report = preflight(root)
    print(json.dumps({"preflight": report}, indent=2))
    if args.preflight_only:
        return
    args.working.mkdir(parents=True, exist_ok=True)
    env = environment()
    run, test_metrics = train(root, args.working)
    archive = package(run, root, report, env, test_metrics, args.working)
    print(json.dumps({"recovery_archive": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
