"""Validate and receipt the V27 Kaggle phone-detector diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "v27_ai85_execution"
RUN_DIR = REPORT / "kaggle_phone_component_v2"
RESULT = RUN_DIR / "v27_phone_component_eval.json"
LOG = RUN_DIR / "roadwatch-v27-phone-component-eval.log"
EXPECTED_CHECKPOINT_SHA256 = (
    "840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54"
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    invariants = {
        "status": "PROVISIONAL_PHONE_COMPONENT_DIAGNOSTIC_COMPLETE",
        "label_status": "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA",
        "selection_eligible": False,
        "production_claim_allowed": False,
        "independent_test_claim": False,
        "training_performed": False,
        "test_split_accessed": False,
        "event_metric_status": "NOT_MEASURED_DIFFERENT_TASK_AND_DATASET",
        "development_ai85_status": "DEVELOPMENT_TARGET_NOT_MET",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "gpu": "Tesla T4",
    }
    for key, expected in invariants.items():
        if result.get(key) != expected:
            raise ValueError(f"KAGGLE_PHONE_RESULT_INVARIANT_FAILED:{key}")
    expected_counts = {
        "images": 1782,
        "phone_positive_images": 181,
        "phone_instances": 229,
    }
    observed_counts = {
        key: result["validation_counts"].get(key) for key in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError("KAGGLE_PHONE_VALIDATION_COUNTS_MISMATCH")

    variants = result["variants"]
    if [row["resolution"] for row in variants] != [640, 960, 1280]:
        raise ValueError("KAGGLE_PHONE_RESOLUTION_SET_MISMATCH")
    if {row["device"] for row in variants} != {"cuda:0"}:
        raise ValueError("KAGGLE_PHONE_NOT_ON_CUDA")
    best = max(variants, key=lambda row: row["f1_from_reported_precision_recall"])
    if best["resolution"] != 960:
        raise ValueError("KAGGLE_PHONE_UNEXPECTED_BEST_RESOLUTION")

    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel": {
            "ref": "longngoc11/roadwatch-v27-phone-component-eval",
            "url": "https://www.kaggle.com/code/longngoc11/roadwatch-v27-phone-component-eval",
            "version": 2,
            "status": "COMPLETE",
        },
        "inputs": {
            "checkpoint_dataset": "longngoc11/roadwatch-ai85-plus-frozen-checkpoints-v8",
            "bootstrap_dataset": "longngoc11/roadwatch-mc-bootstrap-6500-v1",
            "checkpoint_sha256": result["checkpoint_sha256"],
            "bootstrap_manifest_sha256": result["bootstrap_manifest_sha256"],
            "label_status": result["label_status"],
        },
        "downloaded_artifacts": {
            "result": str(RESULT.relative_to(ROOT)),
            "result_sha256": digest(RESULT),
            "log": str(LOG.relative_to(ROOT)),
            "log_sha256": digest(LOG),
        },
        "verified_result": {
            "status": result["status"],
            "scope": result["scope"],
            "gpu": result["gpu"],
            "validation_counts": observed_counts,
            "variants": variants,
            "best_by_f1": best,
            "event_metric_status": result["event_metric_status"],
            "development_ai85_status": result["development_ai85_status"],
            "selection_eligible": result["selection_eligible"],
            "production_claim_allowed": result["production_claim_allowed"],
            "training_performed": result["training_performed"],
            "test_split_accessed": result["test_split_accessed"],
        },
    }
    target = REPORT / "kaggle_phone_component_receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    component_metrics = {
        "status": "PROVISIONAL_COMPONENT_METRICS_ONLY",
        "phone_detector": {
            "label_status": result["label_status"],
            "selection_eligible": False,
            "validation_counts": observed_counts,
            "variants": variants,
            "best_observed_by_f1": best,
        },
        "driver_role_association": {"status": "NOT_ESTABLISHED"},
        "action_classifier": {"status": "NOT_ESTABLISHED"},
        "event_pipeline": {
            "status": "MEASURED_SEPARATELY",
            "result": "reports/v27_ai85_execution/metrics_comparison.json",
        },
    }
    (REPORT / "component_metrics.json").write_text(
        json.dumps(component_metrics, indent=2) + "\n", encoding="utf-8"
    )
    runtime_manifest = {
        "status": "PARTIAL_COMPONENT_RUNTIME_ONLY",
        "environment": {
            key: result[key]
            for key in ("python", "torch", "cuda", "gpu", "ultralytics", "opencv")
        },
        "phone_detector_variants": [
            {
                "resolution": row["resolution"],
                "validator_speed_ms": row["validator_speed_ms"],
                "wall_time_seconds": row["wall_time_seconds"],
                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                "device": row["device"],
            }
            for row in variants
        ],
        "scope_limit": (
            "Ultralytics validator stage timing on 1,782 images; not full-video "
            "decode, association, action, temporal, or end-to-end latency"
        ),
    }
    (REPORT / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": str(target), "best_resolution": 960}, indent=2))


if __name__ == "__main__":
    main()
