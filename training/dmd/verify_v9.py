"""Verify Roadwatch V9 reports, scientific invariants, and optional software checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"
REPORTS = (
    "01_LITERATURE_REVIEW.md",
    "02_DATA_LABEL_QUALITY_AUDIT.md",
    "03_FN_FP_ROOT_CAUSES.md",
    "04_REPRESENTATION_EXPERIMENTS.md",
    "05_FUSION_ABLATIONS.md",
    "06_SEATBELT_EXPERIMENTS.md",
    "07_EVENT_METRICS_COMPARISON.md",
    "08_RUNTIME_BENCHMARK.md",
    "09_TEST_RESULTS.md",
    "10_FINAL_SCIENTIFIC_REPORT.md",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(output: Path, name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path = output / f"{name}.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "seconds": elapsed,
        "log": log_path.relative_to(ROOT).as_posix(),
        "log_sha256": sha256_file(log_path),
    }


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def run_checks(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    checks = [
        _run(output, "compile", [python, "-m", "compileall", "-q", "backend", "training", "tests"]),
        _run(
            output,
            "ruff_critical",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--select",
                "E9,F63,F7,F82",
                "backend",
                "training",
                "tests",
            ],
        ),
        _run(
            output,
            "ruff_v9",
            [
                python,
                "-m",
                "ruff",
                "check",
                "training/dmd/v9_diagnostics.py",
                "training/dmd/render_v9_reports.py",
                "training/dmd/verify_v9.py",
                "tests/test_v9_research.py",
            ],
        ),
    ]
    targeted_xml = output / "pytest_v9.xml"
    checks.append(
        _run(
            output,
            "pytest_v9",
            [
                python,
                "-m",
                "pytest",
                "tests/test_v9_research.py",
                "-q",
                f"--junitxml={targeted_xml}",
            ],
        )
    )
    full_xml = output / "pytest_full.xml"
    checks.append(
        _run(
            output,
            "pytest_full",
            [python, "-m", "pytest", "-q", f"--junitxml={full_xml}"],
        )
    )
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(item["returncode"] == 0 for item in checks) else "FAIL",
        "checks": checks,
        "targeted_pytest": _junit_counts(targeted_xml) if targeted_xml.is_file() else None,
        "full_pytest": _junit_counts(full_xml) if full_xml.is_file() else None,
    }
    _write_json(output / "test_results.json", result)
    return result


def _render_test_report(output: Path, tests: dict[str, Any]) -> None:
    full = tests.get("full_pytest") or {}
    targeted = tests.get("targeted_pytest") or {}
    rows = "\n".join(
        f"| {item['name']} | {'PASS' if item['returncode'] == 0 else 'FAIL'} | "
        f"{item['seconds']:.2f} | `{item['log']}` |"
        for item in tests["checks"]
    )
    text = f"""# V9 test results

Overall status: **{tests["status"]}**.

| Check | Status | Seconds | Evidence |
| --- | --- | --- | --- |
{rows}

- V9 targeted pytest: {targeted.get("tests", "—")} tests, {targeted.get("failures", "—")} failures,
  {targeted.get("errors", "—")} errors, {targeted.get("skipped", "—")} skipped.
- Full pytest: {full.get("tests", "—")} tests, {full.get("failures", "—")} failures,
  {full.get("errors", "—")} errors, {full.get("skipped", "—")} skipped.

These checks validate software behavior and report consistency. They do not establish independent
model accuracy or production readiness.
"""
    (output / "09_TEST_RESULTS.md").write_text(text, encoding="utf-8")


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    diagnostics_path = output / "v9_diagnostics.json"
    tests_path = output / "test_results.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    _render_test_report(output, tests)
    failures = []
    summary = diagnostics["phone_false_negatives"]["summary"]
    if summary["cases"] != 13:
        failures.append("expected exactly 13 frozen FN cases")
    if diagnostics["protected_holdouts_accessed"]:
        failures.append("protected holdout access was reported")
    if diagnostics["ground_truth_changed"] or diagnostics["event_matcher_changed"]:
        failures.append("ground truth or event matcher changed")
    if diagnostics["fine_tuning_performed"] or diagnostics["production_approved"]:
        failures.append("fine-tuning or production approval was claimed")
    if tests["status"] != "PASS":
        failures.append("one or more software checks failed")
    missing = [name for name in REPORTS if not (output / name).is_file()]
    if missing:
        failures.append(f"missing reports: {missing}")
    manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}:
            manifest[path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    _write_json(output / "artifact_manifest.json", manifest)
    validation = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "reports": len([name for name in REPORTS if (output / name).is_file()]),
        "artifacts_hashed": len(manifest),
        "phone_fn_cases": summary["cases"],
        "protected_holdouts_accessed": diagnostics["protected_holdouts_accessed"],
        "fine_tuning_performed": diagnostics["fine_tuning_performed"],
        "production_approved": diagnostics["production_approved"],
        "independent_status": diagnostics["independent_status"],
    }
    _write_json(output / "validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checks", action="store_true")
    args = parser.parse_args()
    if args.checks:
        run_checks(args.output)
    result = verify(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
