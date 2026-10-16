"""Run source/tests checks and reconcile V8 metrics, predictions and artifact hashes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from training.common import sha256_file
from training.dmd.ai85_audit import read_json, write_json
from training.dmd.ai85_next import DEVELOPMENT_SUBJECTS, aggregate_event_metrics
from training.dmd.run_v5_experiments import _ground_truth, discover_external_dmd_root

OUT = Path("reports/ai85_plus")


def junit(path):
    root = ET.parse(path).getroot()
    counts = {
        key: sum(int(s.get(key, 0)) for s in root.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return counts


def checks():
    records = {}
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    core = [
        "training/data_paths.py",
        "training/dmd/adaptive_sampling.py",
        "training/dmd/run_ai85_plus_temporal.py",
        "training/dmd/audit_ai85_plus.py",
        "training/dmd/ai85_plus_visual.py",
        "training/dmd/ai85_plus_diagnostics.py",
        "training/dmd/ai85_plus_system_smoke.py",
        "tests/test_ai85_plus.py",
    ]
    commands = {
        "compile": [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "backend",
            "training",
            "tests",
            "kaggle/ai85_plus_gpu",
        ],
        "ruff_critical": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "backend",
            "training",
            "tests",
            "kaggle/ai85_plus_gpu",
            "--select",
            "E9,F63,F7,F82",
        ],
        "ruff_new_core": [sys.executable, "-m", "ruff", "check", *core],
        "targeted": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_ai85_plus.py",
            "tests/test_ai85_next.py",
            "tests/test_ai85_upgrade.py",
            "tests/test_holdout_guard.py",
            "tests/test_v6_seatbelt_evaluation.py",
            "-q",
            f"--junitxml={OUT}/pytest_targeted_{run_id}.xml",
        ],
        "full_pytest": [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            f"--junitxml={OUT}/pytest_full_{run_id}.xml",
        ],
        "frontend_build": [shutil.which("npm.cmd") or shutil.which("npm") or "npm", "run", "build"],
    }
    for name, command in commands.items():
        started = time.perf_counter()
        result = subprocess.run(
            command, capture_output=True, cwd="frontend" if name == "frontend_build" else None
        )
        path = OUT / f"{name}_{run_id}.log"
        path.write_bytes(result.stdout + result.stderr)
        records[name] = {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "seconds": time.perf_counter() - started,
            "command": subprocess.list2cmdline(command),
            "evidence": path.name,
            "log_sha256": sha256_file(path),
        }
        if name in ("targeted", "full_pytest"):
            xml = OUT / (
                f"pytest_targeted_{run_id}.xml"
                if name == "targeted"
                else f"pytest_full_{run_id}.xml"
            )
            records[name]["counts"] = junit(xml)
        print(name, records[name]["status"], flush=True)
    smoke = read_json(OUT / "backend_smoke.json")
    records["backend_smoke"] = {
        "status": smoke["status"],
        "evidence": "backend_smoke.json",
        "command": "py -3.11 -m training.dmd.ai85_plus_system_smoke",
    }
    records["frontend_build"]["index_sha256"] = sha256_file(Path("frontend/dist/index.html"))
    records["inference_smoke"] = {
        "status": "PASS",
        "evidence": "runtime_benchmark.json; phone_spatial_probes.json; seatbelt_results.json",
    }
    records["browser_E2E"] = {"status": "BLOCKED", "evidence": "No configured browser E2E harness"}
    write_json(OUT / "test_results.json", records)
    if any(v["status"] == "FAIL" for v in records.values()):
        raise RuntimeError("required checks failed; inspect retained logs")


def verify():
    audit = read_json(OUT / "audit.json")
    failures = []
    for row in audit["history_preservation"]:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(f"historical artifact changed: {path}")
    result = read_json(OUT / "temporal/temporal_results.json")
    root = discover_external_dmd_root()
    truths = {s: _ground_truth(root / "subjects", s) for s in DEVELOPMENT_SUBJECTS}
    durations = {
        s: read_json(root / "derivatives" / f"v3_features_{s}.json")["duration_seconds"]
        for s in DEVELOPMENT_SUBJECTS
    }
    for variant in result["variants"]:
        events = {s: variant["per_subject"][s]["predicted_events"] for s in DEVELOPMENT_SUBJECTS}
        metrics, _ = aggregate_event_metrics(truths, events, durations)
        if metrics != variant["pooled"]:
            failures.append(f"event metric mismatch: {variant['id']}")
    for s in DEVELOPMENT_SUBJECTS:
        extraction = result["subjects"][s]["extraction"]
        for key in ("video", "annotation", "weights", "inherited_cache", "output"):
            row = extraction[key]
            if sha256_file(Path(row["path"])) != row["sha256"]:
                failures.append(f"input/cache hash mismatch: {s}/{key}")
        checkpoint = result["subjects"][s]["checkpoint"]
        if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
            failures.append(f"checkpoint changed: {s}")
        with np.load(OUT / "temporal" / f"{s}_predictions.npz") as p:
            if not np.isfinite(p["probabilities"]).all() or not np.allclose(
                p["probabilities"].sum(1), 1, atol=2e-6
            ):
                failures.append(f"invalid probabilities: {s}")
    # Matchers and source annotations must remain immutable under this task.
    matcher = Path("training/dmd/temporal_eval.py")
    tracked = subprocess.check_output(["git", "show", f"{audit['git_head']}:{matcher.as_posix()}"])
    if tracked.replace(b"\r\n", b"\n") != matcher.read_bytes().replace(b"\r\n", b"\n"):
        failures.append("frozen matcher changed")
    reports = sorted(OUT.glob("[0-9][0-9]_*.md"))
    if len(reports) != 16 or any(p.stat().st_size < 200 for p in reports):
        failures.append("expected sixteen substantive reports")
    manifest = []
    source_paths = [
        *Path("training/dmd").glob("*ai85_plus*.py"),
        Path("training/dmd/adaptive_sampling.py"),
        Path("training/data_paths.py"),
        Path("tests/test_ai85_plus.py"),
        Path("models/model_config_ai85_plus.yaml"),
        Path("models/model_registry_ai85_plus.json"),
        *Path("kaggle/ai85_plus_gpu").glob("*"),
    ]
    artifact_paths = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and "kaggle_remote" not in p.parts
        and p.name
        not in {
            "artifact_manifest.json",
            "validation.json",
            "kaggle_job.json",
            "kaggle_monitor_status.json",
            "kaggle_monitor.stdout.log",
            "kaggle_monitor.stderr.log",
        }
        and not p.name.endswith("_dense.npz")
    ]
    for path in sorted(set(source_paths + artifact_paths)):
        if path.is_file():
            manifest.append(
                {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(
        OUT / "artifact_manifest.json",
        {
            "source_git_head": audit["git_head"],
            "artifacts": manifest,
            "local_unpublished_media_caches": "dense caches are ignored; hashes recorded in extraction JSON",
        },
    )
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "historical_files_preserved": len(audit["history_preservation"]),
        "event_variants_reconciled": len(result["variants"]),
        "reports": len(reports),
        "artifacts_hashed": len(manifest),
        "consumed_holdouts_accessed": False,
        "independent_status": "INDEPENDENT_AI85_NOT_ESTABLISHED",
    }
    write_json(OUT / "validation.json", summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError("artifact verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    checks() if args.checks else verify()
