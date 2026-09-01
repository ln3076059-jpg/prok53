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
    parser.add_argument(
        "--dataset-lock", type=Path, default=Path("datasets/manifests/test_frozen.json")
    )
    parser.add_argument("--config", type=Path, default=Path("models/model_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_lock.json"))
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--validation-artifact",
        type=Path,
        default=Path("reports/validation_metrics.json"),
    )
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument(
        "--activation-state",
        choices=["INACTIVE", "CANDIDATE", "ACTIVE"],
        default="INACTIVE",
    )
    args = parser.parse_args()
    if not args.validation_artifact.exists():
        raise ValueError("validation and threshold calibration must precede model lock")
    frozen = json.loads(args.dataset_lock.read_text())
    config = yaml.safe_load(args.config.read_text())
    if config.get("threshold_status") != "CALIBRATED_ON_VALIDATION":
        raise ValueError("thresholds are not marked as validation-calibrated")
    commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        or "UNCOMMITTED"
    )
    calibration = (
        {
            "path": str(args.calibration_artifact),
            "sha256": sha256_file(args.calibration_artifact),
        }
        if args.calibration_artifact and args.calibration_artifact.is_file()
        else {"path": None, "sha256": None, "status": "NOT_PROVIDED"}
    )
    if args.activation_state == "ACTIVE" and calibration["sha256"] is None:
        raise ValueError("ACTIVE model lock requires a threshold calibration artifact")
    component_weight_sha256 = {}
    specialists = config.get("specialized_detectors", {})
    if specialists.get("enabled"):
        for item in specialists.get("models", []):
            component_path = Path(item["weights"])
            if not component_path.is_file():
                raise ValueError(f"specialist weight is missing: {component_path}")
            component_weight_sha256[item["name"]] = sha256_file(component_path)
    lock = {
        "record_schema": "ROADWATCH_MODEL_VERSION_V2",
        "experiment_id": args.experiment_id or config.get("experiment_id", "UNSPECIFIED"),
        "created_at": datetime.now(UTC).isoformat(),
        "locked_at": datetime.now(UTC).isoformat(),
        "activation_state": args.activation_state,
        "weights_sha256": sha256_file(args.weights),
        "component_weight_sha256": component_weight_sha256,
        "config_sha256": sha256_file(args.config),
        "training_data_manifest_sha256": frozen["manifest_sha256"],
        "dataset_sha256": frozen["dataset_sha256"],
        "split_sha256": frozen["manifest_sha256"],
        "validation_metric_artifact": {
            "path": str(args.validation_artifact),
            "sha256": sha256_file(args.validation_artifact),
        },
        "threshold_calibration_artifact": calibration,
        "code_commit": commit,
        "thresholds": config["thresholds"],
        "imgsz": config["imgsz"],
        "class_names": config["class_names"],
        "environment": {
            "ultralytics": ultralytics.__version__,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "production_approved": False,
        "production_approval_status": "NOT_APPROVED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
