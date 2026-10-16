"""Bounded background monitoring of the private frozen-inference Kaggle job."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

OUT = Path("reports/ai85_plus")


def run(maximum_seconds=1800):
    started = time.monotonic()
    source = Path("kaggle/ai85_plus_gpu/run_frozen_gpu.py")
    expected_source = sha256_file(source)
    state = "MONITORING"
    while time.monotonic() - started < maximum_seconds:
        if (
            (OUT / "KAGGLE_MONITOR_STOP").exists()
            or not source.exists()
            or sha256_file(source) != expected_source
        ):
            state = "STOPPED_BY_LOCAL_CHANGE"
            break
        try:
            result = subprocess.run(
                [sys.executable, "-m", "training.dmd.kaggle_ai85_plus"],
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            state = "POLL_TIMEOUT"
            break
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            state = "POLL_ERROR"
            break
        job = json.loads((OUT / "kaggle_job.json").read_text(encoding="utf-8"))
        status = {
            "state": state,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "maximum_monitor_seconds": maximum_seconds,
            "kernel_status": job["status"],
            "kernel": job["kernel"],
        }
        (OUT / "kaggle_monitor_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if job["status"] in {"COMPLETE", "ERROR", "CANCELLED"}:
            state = "RESULT_DOWNLOADED" if job.get("result") else "REMOTE_ENDED_WITHOUT_RESULT"
            break
        time.sleep(45)
    else:
        state = "MONITOR_TIMEOUT_REMOTE_STATUS_RETAINED"
    status = {
        "state": state,
        "pid": os.getpid(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "maximum_monitor_seconds": maximum_seconds,
    }
    (OUT / "kaggle_monitor_status.json").write_text(json.dumps(status, indent=2) + "\n")
    # Rendering consumes only measured JSON; ERROR/timeout does not become a PASS metric.
    for module in ("training.dmd.render_ai85_plus", "training.dmd.verify_ai85_plus"):
        result = subprocess.run([sys.executable, "-m", module], timeout=300)
        if result.returncode:
            raise RuntimeError(f"post-monitor verification failed: {module}")


if __name__ == "__main__":
    run()
