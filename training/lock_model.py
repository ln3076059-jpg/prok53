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


def _configured_component_paths(config: dict) -> dict[str, Path]:
    specialists = config.get("specialized_detectors", {})
    paths = {
        str(item["name"]): Path(item["weights"])
        for item in specialists.get("models", [])
        if item.get("weights")
    }
    auxiliary = config.get("auxiliary_models", {})
    vehicle = config.get("vehicle_context", {})
    optional = {
        "seatbelt_classifier": auxiliary.get("seatbelt_classifier_weights"),
        "pose_estimator": auxiliary.get("pose_weights"),
        "vehicle_detector": vehicle.get("vehicle_weights"),
        "windshield_detector": vehicle.get("cabin_localizer", {}).get("weights"),
    }
    paths.update({name: Path(value) for name, value in optional.items() if value})
    return paths


def validate_calibration_binding(
    calibration_payload: dict,
    locked_model_hashes: dict[str, str],
    configured_thresholds: dict,
) -> None:
    required = {
        "status",
        "created_at",
        "method",
        "model",
        "validation_dataset_manifest",
        "score_artifact",
        "thresholds",
        "classes",
        "test_rows_used",
    }
    missing = required - set(calibration_payload)
    if missing:
        raise ValueError(f"threshold calibration artifact lacks provenance: {sorted(missing)}")
    if calibration_payload["status"] != "CALIBRATED_ON_VALIDATION":
        raise ValueError("threshold calibration artifact is not validation-calibrated")
    if int(calibration_payload["test_rows_used"]) != 0:
        raise ValueError("threshold calibration artifact used frozen test rows")
    calibration_model_hash = calibration_payload.get("model", {}).get("sha256")
    if calibration_model_hash not in set(locked_model_hashes.values()):
        raise ValueError("threshold calibration model SHA256 does not match locked model weights")
    if set(calibration_payload["thresholds"]) != set(configured_thresholds):
        raise ValueError("threshold calibration classes do not match runtime threshold classes")
    for class_name, configured_value in configured_thresholds.items():
        calibrated_value = calibration_payload["thresholds"][class_name]
        if abs(float(configured_value) - float(calibrated_value)) > 1e-12:
            raise ValueError(f"runtime threshold does not match calibration: {class_name}")


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
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite model lock: {args.output}")
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
    primary_weights_sha256 = sha256_file(args.weights)
    component_weight_sha256 = {}
    for name, component_path in _configured_component_paths(config).items():
        if not component_path.is_file():
            raise ValueError(f"configured component weight is missing: {name} ({component_path})")
        component_weight_sha256[name] = sha256_file(component_path)
    if calibration_payload is not None:
        validate_calibration_binding(
            calibration_payload,
            {"primary": primary_weights_sha256, **component_weight_sha256},
            config["thresholds"],
        )
    lock = {
        "record_schema": "ROADWATCH_MODEL_VERSION_V2",
        "experiment_id": args.experiment_id or config.get("experiment_id", "UNSPECIFIED"),
        "created_at": datetime.now(UTC).isoformat(),
        "locked_at": datetime.now(UTC).isoformat(),
        "activation_state": args.activation_state,
        "weights_sha256": primary_weights_sha256,
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
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
