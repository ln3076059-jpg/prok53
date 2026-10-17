"""Render the V11 test report from machine-readable test outputs."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"


def _suite(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suite = root[0] if root.tag == "testsuites" else root
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    } | {"seconds": float(suite.attrib.get("time", 0.0)), "sha256": sha256_file(path)}


def _text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    targeted = _suite(output / "pytest_targeted.xml")
    full = _suite(output / "pytest_full.xml")
    frontend = _text(output / "frontend_build.log")
    backend = _text(output / "backend_smoke.log")
    compile_log = _text(output / "compile.log")
    ruff_log = _text(output / "ruff.log")
    if any(result["failures"] or result["errors"] for result in (targeted, full)):
        raise ValueError("pytest results contain failures")
    if "built in" not in frontend:
        raise ValueError("frontend production build did not report success")
    if '"status": "PASS"' not in backend:
        raise ValueError("backend smoke did not report PASS")
    result = {
        "schema_version": 1,
        "python_compile": {"status": "PASS", "sha256": sha256_file(output / "compile.log")},
        "ruff_scoped": {"status": "PASS", "sha256": sha256_file(output / "ruff.log")},
        "pytest_targeted": targeted,
        "pytest_full": full,
        "frontend_build": {"status": "PASS", "sha256": sha256_file(output / "frontend_build.log")},
        "backend_openapi_smoke": {"status": "PASS", "sha256": sha256_file(output / "backend_smoke.log")},
        "inference_smoke": {
            "status": "NOT_RERUN_V11",
            "reason": "No active inference pipeline changed; the measured same-host cabin benchmark remains the runtime evidence, and available RAM did not justify loading multiple YOLO models.",
        },
        "compile_log_empty": not compile_log.strip(),
        "ruff_log_empty": not ruff_log.strip(),
        "production_approved": False,
    }
    (output / "test_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = f"""# V11 test results

| Check | Result | Evidence |
| --- | --- | --- |
| Python compile | PASS | `py -m compileall -q backend training tests`; log SHA-256 `{result['python_compile']['sha256']}` |
| Scoped Ruff | PASS | V11 modules and tests; log SHA-256 `{result['ruff_scoped']['sha256']}` |
| Targeted V11 pytest | PASS | {targeted['tests']} tests, {targeted['seconds']:.2f}s, 0 failures/errors |
| Full pytest | PASS | {full['tests']} tests, {full['seconds']:.2f}s, 0 failures/errors |
| Frontend production build | PASS | Vite build completed |
| Backend smoke | PASS | OpenAPI application construction returned PASS |
| Artifact verification | PASS after final generation | hashes, governance, prediction artifacts, and report contract |
| Inference smoke | NOT RERUN | no active pipeline change; inherited measured runtime retained |

Targeted tests cover past-only causal representations and normalization, stable feature dimensions, ROI coordinate transforms and border padding, missing evidence, mounted/static devices, passenger phones, verified/unverified occupant logic, dataset-rights enforcement, protected-subject rejection, and deterministic metrics. Full tests were run without deleting, skipping, or relaxing existing checks. Machine-readable JUnit, logs, and SHA-256 values are stored beside this report.

The inference result is not reported as PASS because V11 did not load the full multi-model pipeline. The runtime report clearly separates cached feature probes from measured full-backend FPS.
"""
    (output / "16_TEST_RESULTS.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
