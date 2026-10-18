"""Verify the V17 evidence bundle and executed checks."""

# ruff: noqa: E501

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v17_ai85_recovery"
REPORTS = [f"0{i}_{name}.md" for i, name in enumerate([
    "DATA_ACCESS_ACTION_PLAN",
    "ANNOTATION_BENCHMARK_AND_SPLIT",
    "SAHI_AND_ACTION_RESULTS",
    "ASSOCIATION_TEMPORAL_AND_SEATBELT",
    "METRICS_AND_TESTS",
    "FINAL_SCIENTIFIC_DECISION",
], start=1)]
ARTIFACTS = [
    "dataset_access_action_plan.json", "approved_data_inventory.json",
    "phone_benchmark_protocol.json", "annotation_qa.json", "group_split_manifest.json",
    "sahi_ablation.json", "action_experiments.json", "association_fusion_analysis.json",
    "temporal_event_analysis.json", "event_metrics.json", "seatbelt_review_status.json",
    "experiment_registry.json", "model_manifest.json", "environment.json", "reproduce.ps1",
]


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_log(name: str) -> str:
    payload = (OUT / name).read_bytes()
    return payload.decode("utf-16" if payload.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8")


def junit(name: str) -> dict:
    root = ET.parse(OUT / name).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {key: sum(int(float(s.attrib.get(key, 0))) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    result["status"] = "PASS" if result["tests"] and not result["failures"] and not result["errors"] else "FAIL"
    return result


def verify() -> None:
    missing = [name for name in REPORTS + ARTIFACTS if not (OUT / name).is_file()]
    if missing:
        raise ValueError(f"missing artifacts: {missing}")
    inventory, benchmark, split = (read(name) for name in ("approved_data_inventory.json", "phone_benchmark_protocol.json", "group_split_manifest.json"))
    experiments, models, events = (read(name) for name in ("experiment_registry.json", "model_manifest.json", "event_metrics.json"))
    targeted, full = junit("pytest_targeted.xml"), junit("pytest_full.xml")
    results = {"targeted": targeted, "full": full, "compile": "PASS" if "PASS" in read_log("compile.log") else "FAIL", "ruff": "PASS" if "All checks passed" in read_log("ruff.log") else "FAIL", "frontend": "PASS" if "built in" in read_log("frontend_build.log") else "FAIL", "backend": "PASS" if '"status": "PASS"' in read_log("backend_smoke.log") else "FAIL"}
    checks = {"zero_approved_rows": inventory["approved_rows"] == 0, "no_protected_access": not inventory["protected_subjects_accessed"], "benchmark_blocked": benchmark["ap_metrics_allowed"] is False, "no_independent_split": split["independent"] is False, "no_training": experiments["training_performed"] is False, "no_model_change": models["active_model_changed"] is False, "event_deltas_zero": all(events[key] == 0 for key in ("delta_tp", "delta_fp", "delta_fn")), "executed_checks_pass": targeted["status"] == full["status"] == "PASS" and all(value == "PASS" for key, value in results.items() if key not in {"targeted", "full"})}
    status = "PASS" if all(checks.values()) else "FAIL"
    (OUT / "test_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    validation = {"status": status, "checks": checks, "independent_ai85_established": False, "production_approved": False}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = REPORTS + ARTIFACTS + ["compile.log", "ruff.log", "pytest_targeted.log", "pytest_targeted.xml", "pytest_full.log", "pytest_full.xml", "frontend_build.log", "backend_smoke.log", "test_results.json", "validation.json"]
    (OUT / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "files": {name: {"bytes": (OUT / name).stat().st_size, "sha256": sha256_file(OUT / name)} for name in files}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise ValueError(json.dumps(validation, sort_keys=True))
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    verify()
