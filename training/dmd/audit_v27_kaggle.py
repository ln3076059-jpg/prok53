"""Create a reproducible receipt for the completed V27 Kaggle GPU preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "v27_ai85_execution"
RESULT = REPORT / "kaggle_gpu_preflight_v3" / "v27_gpu_preflight.json"
LOG = REPORT / "kaggle_gpu_preflight_v3" / "roadwatch-v27-gpu-preflight.log"
MANIFEST = ROOT / "kaggle" / "roadwatch_v27_ai85_execution" / "package-manifest.json"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    required = {
        "status": "V27_GPU_PREFLIGHT_PASSED",
        "event_metrics_computed": False,
        "dmd_media_used": False,
        "candidate_identity_test": "PASSED",
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"KAGGLE_RESULT_INVARIANT_FAILED:{key}")
    if set(result["device_provenance"].values()) != {"cuda:0"}:
        raise ValueError("KAGGLE_MODELS_NOT_ALL_ON_CUDA")
    manifest_sha = digest(MANIFEST)
    if result["code_manifest_sha256"] != manifest_sha:
        raise ValueError("KAGGLE_CODE_MANIFEST_HASH_MISMATCH")

    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "ref": "longngoc11/roadwatch-v27-ai85-execution",
            "url": "https://www.kaggle.com/datasets/longngoc11/roadwatch-v27-ai85-execution",
            "visibility": "private",
            "version_used": 3,
            "code_manifest_sha256": manifest_sha,
            "contains_dmd_media": False,
            "contains_credentials": False,
        },
        "kernel": {
            "ref": "longngoc11/roadwatch-v27-gpu-preflight",
            "url": "https://www.kaggle.com/code/longngoc11/roadwatch-v27-gpu-preflight",
            "version": 3,
            "status": "COMPLETE",
        },
        "downloaded_artifacts": {
            "result": str(RESULT.relative_to(ROOT)),
            "result_sha256": digest(RESULT),
            "log": str(LOG.relative_to(ROOT)),
            "log_sha256": digest(LOG),
        },
        "verified_result": {
            key: result[key]
            for key in (
                "status",
                "scope",
                "training_performed",
                "dmd_media_used",
                "event_metrics_computed",
                "candidate_identity_test",
                "gpu",
                "device_provenance",
                "stage_runtime",
                "peak_cuda_allocated_bytes",
                "raw_d5_status",
                "development_ai85_status",
            )
        },
    }
    target = REPORT / "kaggle_execution_receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(target), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
