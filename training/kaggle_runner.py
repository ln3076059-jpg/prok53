from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image

CANONICAL_NAMES = ["phone", "seatbelt_fastened", "seatbelt_unfastened"]
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
    candidates = list(Path("/kaggle/input").rglob("expected_hashes.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one MC_001 bundle, found {len(candidates)}")
    return candidates[0].parent


def verify_hashes(root: Path) -> int:
    expected = json.loads((root / "expected_hashes.json").read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"STOP: bundle hash mismatch: {relative}")
    return len(expected)


def read_names(data: dict) -> list[str]:
    names = data.get("names", {})
    return list(names) if isinstance(names, list) else [names[index] for index in sorted(names)]


def preflight(root: Path) -> dict:
    verified = verify_hashes(root)
    dataset = root / "dataset"
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if data.get("nc") != 3 or read_names(data) != CANONICAL_NAMES:
        raise ValueError("STOP: canonical class order mismatch")
    split_metadata = json.loads((dataset / "split_metadata.json").read_text(encoding="utf-8"))
    if split_metadata.get("isolation", {}).get("status") != "PASS":
        raise ValueError("STOP: group/split isolation did not pass")
    frozen = json.loads((root / "test_frozen.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise ValueError("STOP: test split is not frozen")
    splits = {}
    for split in ("train", "val", "test"):
        counts = Counter()
        images = [path for path in (dataset / "images" / split).glob("*") if path.is_file()]
        if not images:
            raise ValueError(f"STOP: empty split: {split}")
        for image_path in images:
            with Image.open(image_path) as image:
                image.verify()
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"STOP: missing label: {label_path.relative_to(root)}")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) not in (0, 1, 2):
                    raise ValueError(f"STOP: invalid label: {label_path.relative_to(root)}")
                counts[int(parts[0])] += 1
        missing_classes = [CANONICAL_NAMES[index] for index in range(3) if counts[index] == 0]
        if missing_classes:
            raise ValueError(f"STOP: {split} lacks classes: {missing_classes}")
        splits[split] = {"images": len(images), "instances": {CANONICAL_NAMES[i]: counts[i] for i in range(3)}}
    return {"verified_files": verified, "splits": splits, "frozen_test_samples": frozen["test_samples"]}


def write_runtime_data_yaml(root: Path, working: Path) -> Path:
    data = yaml.safe_load((root / "dataset" / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str((root / "dataset").resolve())
    target = working / "mc001_data.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


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
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout.splitlines(),
    }


def train(root: Path, working: Path) -> Path:
    from ultralytics import YOLO

    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    unknown = set(config) - TRAIN_KEYS - {"experiment_id", "model", "data"}
    if unknown:
        raise ValueError(f"STOP: unapproved training keys: {sorted(unknown)}")
    model_name = config.pop("model")
    config.pop("experiment_id", None)
    config.pop("data", None)
    config = {key: value for key, value in config.items() if key in TRAIN_KEYS}
    config["data"] = str(write_runtime_data_yaml(root, working))
    config["project"] = str(working)
    config["name"] = "MC_001"
    config["exist_ok"] = False
    YOLO(model_name).train(**config)
    return working / "MC_001"


def package(run: Path, root: Path, preflight_report: dict, env: dict, working: Path) -> Path:
    required = ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"]
    for relative in required:
        if not (run / relative).is_file():
            raise FileNotFoundError(f"missing training artifact: {relative}")
    metadata = {
        "experiment_id": "MC_001",
        "bundle_repository_commit": (root / "repository_commit.txt").read_text(encoding="utf-8").strip(),
        "best_weights_sha256": sha256_file(run / "weights" / "best.pt"),
        "preflight": preflight_report,
        "environment": env,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    archive_base = working / "MC_001_RECOVERY"
    return Path(shutil.make_archive(str(archive_base), "zip", run))


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical fail-closed Kaggle MC_001 runner")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--working", type=Path, default=Path(os.environ.get("MC001_WORKING", "/kaggle/working")))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = locate_bundle(args.bundle)
    report = preflight(root)
    print(json.dumps({"preflight": report}, indent=2))
    if args.preflight_only:
        return
    env = environment()
    args.working.mkdir(parents=True, exist_ok=True)
    run = train(root, args.working)
    archive = package(run, root, report, env, args.working)
    print(json.dumps({"recovery_archive": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
