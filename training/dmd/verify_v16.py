"""Validate the V16 report bundle and emit machine-readable receipts."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v16_data_ai85"
REPORTS = [
    "01_V15_CODE_AUDIT.md",
    "02_DATASET_RESEARCH_AND_LICENSES.md",
    "03_DATASET_SELECTION_MATRIX.md",
    "04_DATA_INGESTION_AND_QA.md",
    "05_ANNOTATION_AND_SPLIT_PROTOCOL.md",
    "06_PHONE_DETECTION_RESULTS.md",
    "07_ACTION_RECOGNITION_RESULTS.md",
    "08_ASSOCIATION_FUSION_ANALYSIS.md",
    "09_TEMPORAL_EVENT_RESULTS.md",
    "10_SEATBELT_RESULTS.md",
    "11_END_TO_END_METRICS.md",
    "12_RUNTIME_AND_TESTS.md",
    "13_FINAL_AI_LEAD_DECISION.md",
]
ARTIFACTS = [
    "dataset_registry.json",
    "license_audit.json",
    "data_manifest.json",
    "label_mapping.json",
    "annotation_qa.json",
    "group_split_manifest.json",
    "experiment_registry.json",
    "metrics_comparison.json",
    "model_manifest.json",
    "candidate_readiness.json",
    "environment.json",
    "report_evidence_hashes.json",
    "reproduce.ps1",
]


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_log(name: str) -> str:
    payload = (OUT / name).read_bytes()
    encoding = "utf-16" if payload.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    return payload.decode(encoding)


def junit(name: str) -> dict:
    root = ET.parse(OUT / name).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    summary["status"] = (
        "PASS" if summary["tests"] and not summary["failures"] and not summary["errors"] else "FAIL"
    )
    return summary


def verify() -> dict:
    missing = [name for name in REPORTS + ARTIFACTS if not (OUT / name).is_file()]
    if missing:
        raise ValueError(f"missing V16 artifacts: {missing}")

    registry = read("dataset_registry.json")
    manifest = read("data_manifest.json")
    split = read("group_split_manifest.json")
    experiments = read("experiment_registry.json")
    models = read("model_manifest.json")
    checks = {
        "nine_dataset_candidates": len(registry["registry"]) == 9,
        "zero_v16_downloads": registry["v16_downloads_performed"] == 0,
        "zero_approved_training_records": manifest["total_approved_for_training"] == 0,
        "protected_holdouts_not_accessed": not manifest["protected_subjects_accessed"],
        "no_independent_split_claim": split["independent_evaluation"] is False,
        "no_training": experiments["training_performed"] is False,
        "no_active_model_change": models["active_model_changed"] is False,
        "no_production_approval": models["production_approved"] is False,
    }
    for model in (models["phone_detector"], models["auxiliary_coco"]):
        path = ROOT / model["path"]
        checks[f"model_hash:{model['path']}"] = (
            path.is_file() and sha256_file(path) == model["sha256"]
        )

    targeted = junit("pytest_targeted.xml")
    full = junit("pytest_full.xml")
    test_results = {
        "targeted": targeted,
        "full": full,
        "compile": "PASS" if "PASS" in read_log("compile.log") else "FAIL",
        "ruff": "PASS" if "All checks passed" in read_log("ruff.log") else "FAIL",
        "frontend": "PASS" if "built in" in read_log("frontend_build.log") else "FAIL",
        "backend": (
            "PASS"
            if '"status": "PASS"' in read_log("backend_smoke.log")
            else "FAIL"
        ),
    }
    (OUT / "test_results.json").write_text(
        json.dumps(test_results, indent=2) + "\n", encoding="utf-8"
    )
    checks["all_executed_checks_pass"] = all(
        value == "PASS" for key, value in test_results.items() if key not in {"targeted", "full"}
    ) and targeted["status"] == full["status"] == "PASS"
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "reports": len(REPORTS),
        "datasets_audited": len(registry["registry"]),
        "training_performed": False,
        "independent_ai85_established": False,
        "production_approved": False,
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = REPORTS + ARTIFACTS + [
        "compile.log",
        "ruff.log",
        "pytest_targeted.log",
        "pytest_targeted.xml",
        "pytest_full.log",
        "pytest_full.xml",
        "frontend_build.log",
        "backend_smoke.log",
        "test_results.json",
        "validation.json",
    ]
    artifact_manifest = {
        "schema_version": 1,
        "files": {
            name: {"bytes": (OUT / name).stat().st_size, "sha256": sha256_file(OUT / name)}
            for name in names
        },
    }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "PASS":
        raise ValueError(json.dumps(validation, sort_keys=True))
    print(json.dumps(validation, indent=2, sort_keys=True))
    return validation


if __name__ == "__main__":
    verify()
