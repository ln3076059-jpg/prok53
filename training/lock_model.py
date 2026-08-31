from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
import ultralytics
import yaml

from training.common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--dataset-lock", type=Path, default=Path("datasets/manifests/test_frozen.json"))
    parser.add_argument("--config", type=Path, default=Path("models/model_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_lock.json"))
    args = parser.parse_args()
    if not Path("reports/validation_metrics.json").exists():
        raise ValueError("validation and threshold calibration must precede model lock")
    frozen = json.loads(args.dataset_lock.read_text())
    config = yaml.safe_load(args.config.read_text())
    if config.get("threshold_status") != "CALIBRATED_ON_VALIDATION":
        raise ValueError("thresholds are not marked as validation-calibrated")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "UNCOMMITTED"
    lock = {"locked_at": datetime.now(UTC).isoformat(), "weights_sha256": sha256_file(args.weights), "dataset_sha256": frozen["dataset_sha256"], "split_sha256": frozen["manifest_sha256"], "code_commit": commit, "thresholds": config["thresholds"], "imgsz": config["imgsz"], "class_names": config["class_names"], "ultralytics": ultralytics.__version__, "pytorch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

