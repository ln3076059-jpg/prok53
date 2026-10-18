"""Fail-closed verification for V14 evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v14_research"
REPORTS = [f"{i:02d}_{name}.md" for i, name in enumerate((
    "HEAD_AND_RIGHTS_AUDIT", "V13_BASELINE_RECONCILIATION", "AUTONOMOUS_PHONE_PROPOSALS",
    "HIGH_RES_BEHAVIOR_PIPELINE", "ONLINE_E3_MODEL_READINESS", "DRIVER_ASSOCIATION_EVIDENCE",
    "EVENT_REPLAY_RESULTS", "SEATBELT_READINESS", "RUNTIME_AND_TESTS",
    "FINAL_AI_LEAD_DECISION",
), 1)]


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
    auto, data = read("autonomous_proposal_audit.json"), read("data_manifest.json")
    ready = read("candidate_readiness.json")
    metrics = read("metrics_comparison.json")
    tests = read("test_results.json")
    if auto["reviewed_boxes_used_as_input"] or auto["input_phone_boxes"] != "DETECTOR_RUNTIME_ONLY":
        raise ValueError("autonomous/oracle boundary violated")
    if any(auto[key] is not None for key in ("phone_ap", "precision", "proposal_recall")):
        raise ValueError("unsupported detector accuracy claim")
    rights_closed = data["live_permissions"]["RIGHTS_ACCEPTED"] == "false"
    if not rights_closed or data["protected_media_or_labels_opened"]:
        raise ValueError("rights or holdout invariant failed")
    if ready["production"] != "REJECT" or metrics["v14_new_event_metrics"] is not None:
        raise ValueError("unsupported promotion or event claim")
    if tests["full"]["tests"] < 622 or tests["full"]["failures"] or tests["full"]["errors"]:
        raise ValueError("full tests incomplete")
    if any(tests[key] != "PASS" for key in ("compile", "ruff", "frontend", "backend")):
        raise ValueError("engineering check failed")
    result = {
        "status":"PASS", "reports":10, "autonomous_adapter":"READY_ENGINEERING",
        "autonomous_accuracy":"INCONCLUSIVE_NO_COMPLETE_GROUND_TRUTH",
        "training_performed":False, "protected_holdouts_accessed":False,
        "independent_ai85_established":False, "production_approved":False,
        "final_status":"RESEARCH_ONLY",
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
