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
        "--readiness-artifact",
        type=Path,
        default=Path("reports/v2_training_readiness.json"),
    )
    parser.add_argument("--fusion-artifact", type=Path)
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
    calibration_payload = None
    if calibration["sha256"] is not None:
        calibration_payload = json.loads(args.calibration_artifact.read_text(encoding="utf-8"))
        required_calibration_fields = {
            "status",
            "model",
            "validation_dataset_manifest",
            "score_artifact",
            "test_rows_used",
        }
        missing = required_calibration_fields - set(calibration_payload)
        if missing:
            raise ValueError(f"threshold calibration artifact lacks provenance: {sorted(missing)}")
        if calibration_payload["status"] != "CALIBRATED_ON_VALIDATION":
            raise ValueError("threshold calibration artifact is not validation-calibrated")
        if int(calibration_payload["test_rows_used"]) != 0:
            raise ValueError("threshold calibration artifact used frozen test rows")

    readiness = None
    if args.readiness_artifact.is_file():
        readiness = json.loads(args.readiness_artifact.read_text(encoding="utf-8"))
    if args.activation_state == "ACTIVE" and not (
        readiness and readiness.get("status", {}).get("governed_training_ready") is True
    ):
        raise ValueError("ACTIVE model lock requires governed human-review readiness")

    fusion_path = args.fusion_artifact
    if fusion_path is None:
        configured_fusion = config.get("auxiliary_models", {}).get("fusion_artifact")
        fusion_path = Path(configured_fusion) if configured_fusion else None
    fusion_record = {"path": None, "sha256": None, "status": "NOT_CONFIGURED"}
    fusion_required = bool(config.get("activation_policy", {}).get("require_calibrated_fusion"))
    if fusion_path and fusion_path.is_file():
        fusion_payload = json.loads(fusion_path.read_text(encoding="utf-8"))
        required_fusion_fields = {
            "feature_names",
            "means",
            "scales",
            "weights",
            "intercept",
            "threshold",
            "development_manifest",
            "validation",
            "test_rows_used",
        }
        missing = required_fusion_fields - set(fusion_payload)
        if missing:
            raise ValueError(f"fusion artifact lacks provenance/schema: {sorted(missing)}")
        if int(fusion_payload["test_rows_used"]) != 0:
            raise ValueError("fusion artifact used frozen test rows")
        fusion_record = {
            "path": str(fusion_path),
            "sha256": sha256_file(fusion_path),
            "status": fusion_payload.get("status", "SCHEMA_VALID"),
        }
    elif args.activation_state == "ACTIVE" and fusion_required:
        raise ValueError("ACTIVE model lock requires the configured calibrated fusion artifact")
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
        "human_review_readiness_artifact": {
            "path": str(args.readiness_artifact),
            "sha256": sha256_file(args.readiness_artifact) if readiness else None,
            "governed_training_ready": (
                readiness.get("status", {}).get("governed_training_ready") if readiness else False
            ),
        },
        "fusion_artifact": fusion_record,
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
