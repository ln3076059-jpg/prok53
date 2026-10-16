"""Inspect/fetch the already-authorized private V8 Kaggle inference job."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

REFERENCE = "longngoc11/roadwatch-ai85-plus-frozen-gpu-v8"
OUT = Path("reports/ai85_plus")


def run():
    command = [sys.executable, "-m", "kaggle", "kernels", "status", REFERENCE]
    response = subprocess.run(command, capture_output=True, text=True, timeout=90)
    message = re.sub(r"KGAT_[A-Za-z0-9_-]+", "[REDACTED]", response.stdout + response.stderr)
    status = next(
        (
            v
            for v in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED")
            if f"KernelWorkerStatus.{v}" in message
        ),
        "UNKNOWN",
    )
    record = {
        "kernel": REFERENCE,
        "version": 1,
        "private": True,
        "status": status,
        "url": "https://www.kaggle.com/code/" + REFERENCE,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "status_message": message.strip(),
        "source_file": "kaggle/ai85_plus_gpu/run_frozen_gpu.py",
        "source_sha256": sha256_file(Path("kaggle/ai85_plus_gpu/run_frozen_gpu.py")),
        "training_performed": False,
        "DMD_uploaded": False,
        "holdouts_attached": False,
        "credential_storage": "restricted user-profile file outside the repository",
    }
    if status in {"COMPLETE", "ERROR"}:
        destination = OUT / "kaggle_gpu/version1"
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "gpu_frozen_results.json"
        if not result_path.exists():
            fetched = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "kernels",
                    "output",
                    REFERENCE,
                    "-p",
                    str(destination.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            record["download_returncode"] = fetched.returncode
            record["download_message"] = fetched.stdout + fetched.stderr
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            expected = "840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54"
            if result["weights_sha256"] != expected or result["holdouts_accessed"]:
                raise ValueError("GPU result input/checkpoint governance mismatch")
            record["result"] = {
                "path": result_path.as_posix(),
                "sha256": sha256_file(result_path),
                "gpu": result["gpu"],
                "measured_variants": len(result["variants"]),
            }
    (OUT / "kaggle_job.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    run()
