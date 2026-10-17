"""Fail-closed verification of V13 research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v13_research"
REPORTS = [
    "01_HEAD_AND_ENVIRONMENT_AUDIT.md", "02_DATA_RIGHTS_AND_SPLIT_AUDIT.md",
    "03_BASELINE_RECONCILIATION.md", "04_PHONE_HIGH_RES_ROI.md",
    "05_ONLINE_E3_ARTIFACT_READINESS.md", "06_BEHAVIOR_REPRESENTATION_RESULTS.md",
    "07_DRIVER_PHONE_ASSOCIATION.md", "08_FUSION_AND_EVENT_REPLAY.md",
    "09_SEATBELT_RESULTS.md", "10_RUNTIME_BENCHMARK.md", "11_TEST_RESULTS.md",
    "12_FINAL_AI_LEAD_REPORT.md",
]


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def verify() -> dict:
    missing = [name for name in REPORTS if not (OUT / name).is_file()]
    small = [
        name
        for name in REPORTS
        if (OUT / name).is_file() and (OUT / name).stat().st_size < 500
    ]
    if missing or small:
        raise ValueError(f"missing={missing}, undersized={small}")
    data, roi = read("dataset_manifest.json"), read("high_resolution_roi.json")
    ready, metrics = read("candidate_readiness.json"), read("metrics_comparison.json")
    tests, events = read("test_results.json"), read("event_match_records.json")
    if data["permissions"]["RIGHTS_ACCEPTED"] != "false":
        raise ValueError("unexpected rights state")
    if data["protected_media_or_labels_opened"]:
        raise ValueError("protected holdout accessed")
    population = (roi["decoded_frames"], roi["reviewed_boxes"], roi["r5_full_phone_coverage"])
    if population != (13, 12, 12):
        raise ValueError("high-resolution ROI population mismatch")
    if roi["phone_ap"] is not None or roi["event_recovery_claimed"]:
        raise ValueError("unsupported ROI accuracy claim")
    if metrics["v13_new_accuracy_metrics"] is not None or metrics["independent_ai85_established"]:
        raise ValueError("unsupported accuracy claim")
    if ready["production"] != "REJECT" or not ready["event_replay"].startswith("BLOCKED"):
        raise ValueError("candidate readiness is unsafe")
    if events["matcher_changed"] or events["v13_candidate_replay"].split("_")[0] != "BLOCKED":
        raise ValueError("event evidence policy changed")
    if any(tests[key] != "PASS" for key in ("compile", "ruff", "frontend", "backend")):
        raise ValueError("engineering check failed")
    if tests["full"]["failures"] or tests["full"]["errors"] or tests["full"]["tests"] < 619:
        raise ValueError("full test suite incomplete")
    result = {
        "status": "PASS", "reports": len(REPORTS), "decoded_original_frames": 13,
        "reviewed_boxes_fully_covered": 12, "training_performed": False,
        "protected_holdouts_accessed": False, "independent_ai85_established": False,
        "production_approved": False, "final_status": "RESEARCH_ONLY",
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
