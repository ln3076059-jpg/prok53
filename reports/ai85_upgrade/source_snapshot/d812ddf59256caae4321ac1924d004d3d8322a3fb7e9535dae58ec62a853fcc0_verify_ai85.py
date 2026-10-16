"""Run and record actual checks without concealing inherited suite failures."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT = Path("reports/ai85_upgrade")
TARGETS = [
    "training/dmd/ai85_action.py", "training/dmd/ai85_audit.py",
    "training/dmd/ai85_detector_probe.py", "training/dmd/ai85_native_embeddings.py",
    "training/dmd/run_ai85_upgrade.py", "training/dmd/render_ai85_reports.py",
    "training/dmd/verify_ai85.py", "tests/test_ai85_upgrade.py",
]


def run_check(name: str, command: list[str]) -> dict:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    log = OUTPUT / f"{name}.log"
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    row = {"command": command, "exit_code": result.returncode,
           "status": "PASS" if result.returncode == 0 else "FAIL",
           "seconds": time.perf_counter() - started, "log": log.as_posix()}
    print(f"{name}: {row['status']} ({row['seconds']:.1f}s)", flush=True)
    return row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {}
    commands = {
        "ruff": [sys.executable, "-m", "ruff", "check", *TARGETS],
        "compile": [sys.executable, "-m", "compileall", "-q", *TARGETS],
        "pytest_full": [sys.executable, "-m", "pytest", "tests", "-q",
                        "--continue-on-collection-errors", "--junitxml",
                        str(OUTPUT / "pytest_full.xml")],
        "pytest_targeted": [sys.executable, "-m", "pytest", "-q",
                            "tests/test_ai85_upgrade.py", "tests/test_v7_action_models.py",
                            "tests/test_v7_result_integrity.py",
                            "tests/test_v6_action_hierarchy.py",
                            "tests/test_v6_seatbelt_evaluation.py", "tests/test_v5_video_action.py",
                            "tests/test_holdout_guard.py", "tests/test_engineering_freeze.py",
                            "tests/test_runtime_v2.py", "tests/test_v4_experiment_runner.py",
                            "--junitxml", str(OUTPUT / "pytest_targeted.xml")],
    }
    for name, command in commands.items():
        result[name] = run_check(name, command)
        xml_path = OUTPUT / f"{name}.xml"
        if xml_path.exists():
            tree = ET.parse(xml_path)
            suites = tree.getroot().findall("testsuite")
            result[name]["counts"] = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                                      for key in ("tests", "errors", "failures", "skipped")}
            result[name]["failures"] = [
                {"test": c.attrib, "kind": child.tag,
                 "message": child.attrib.get("message", ""), "details": child.text}
                for s in suites for c in s.findall("testcase") for child in c
                if child.tag in ("error", "failure")
            ]
        (OUTPUT / "test_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
