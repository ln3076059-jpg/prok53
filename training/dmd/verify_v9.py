"""Verify Roadwatch V9 reports, scientific invariants, and optional software checks."""

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
    "11_PHONE_PROPOSAL_MINING.md",
    "12_HUMAN_REVIEW_WEB.md",
    "13_HUMAN_PHONE_REVIEW_RESULTS.md",
    "14_PHONE_CONTEXT_GATE_ABLATION.md",
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
    npm = shutil.which("npm") or "npm"
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
                "training/dmd/v9_phone_proposal_mining.py",
                "training/dmd/v9_human_review_results.py",
                "training/dmd/v9_phone_context_ablation.py",
                "training/dmd/fusion.py",
                "backend/services/research_review.py",
                "backend/schemas.py",
                "tests/test_v9_research.py",
                "tests/test_v3_fusion.py",
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
    checks.append(_run(output, "frontend_build", [npm, "--prefix", "frontend", "run", "build"]))
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
    proposals_path = output / "phone_proposal_mining.json"
    tests_path = output / "test_results.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    human_review = json.loads(
        (output / "human_phone_review_summary.json").read_text(encoding="utf-8")
    )
    context_ablation = json.loads(
        (output / "phone_context_gate_ablation.json").read_text(encoding="utf-8")
    )
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
    if proposals["summary"]["cases"] != 13:
        failures.append("phone proposal mining did not preserve the 13-case FN set")
    if proposals["summary"]["verified_occupant_tracks"] != 0:
        failures.append("phone proposal mining claimed an unverified occupant track")
    if human_review["summary"]["human_reviewed"] != 13:
        failures.append("human phone review is incomplete")
    if human_review["summary"]["valid_phone_proposals"] != 12:
        failures.append("unexpected human-reviewed phone proposal count")
    if human_review["summary"]["no_valid_proposal_hard_examples"] != 1:
        failures.append("visible-phone hard-example count is invalid")
    if human_review["governance"]["training_admission"] != (
        "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY"
    ):
        failures.append("human review lane bypasses development-only governance")
    context_floor = context_ablation["mining_floor_result"]
    if context_ablation["metric_semantics"] != (
        "CANDIDATE_YIELD_NOT_ACCURACY_OR_EVENT_METRICS"
    ):
        failures.append("phone-context ablation overclaims metric semantics")
    if context_ablation["proposal_summary"]["proposals"] != 226:
        failures.append("phone-context ablation proposal set changed")
    if context_floor["removed_proposals"] != 99:
        failures.append("phone-context gate mounted/static removal count changed")
    if context_floor["human_selected_retained_context_gated"] != 12:
        failures.append("phone-context gate lost a human-selected proposal at mining floor")
    if context_floor["hard_example_has_context_gated_candidate"]:
        failures.append("phone-context gate admitted the mounted-camera hard example")
    if any(
        context_ablation["governance"][key]
        for key in (
            "protected_holdouts_accessed",
            "ground_truth_changed",
            "event_matcher_changed",
            "fine_tuning_performed",
            "production_approved",
        )
    ):
        failures.append("phone-context ablation changed a protected scientific invariant")
    if any(
        proposals[key]
        for key in (
            "ground_truth_changed",
            "event_matcher_changed",
            "fine_tuning_performed",
            "production_approved",
        )
    ):
        failures.append("phone proposal mining changed a protected scientific invariant")
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
        "context_gate_removed_proposals": context_floor["removed_proposals"],
        "context_gate_selected_retained": context_floor[
            "human_selected_retained_context_gated"
        ],
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
