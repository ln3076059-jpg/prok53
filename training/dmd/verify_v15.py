"""Fail-closed validation of the V15 research bundle."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v15_research"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def run() -> None:
    validation = load("validation.json")
    assert validation["reports_present"] is True
    assert validation["experiment_count"] <= 3
    assert validation["training_performed"] is False
    assert validation["protected_holdouts_accessed"] is False
    assert validation["production_approved"] is False
    assert validation["independent_ai85_established"] is False
    assert load("dataset_manifest.json")["permissions"]["RIGHTS_ACCEPTED"] == "false"
    assert load("metrics_comparison.json")["v15_new_accuracy_metrics"] is None
    assert load("sliced_phone_diagnostic.json")["model_inference_included"] is False
    assert load("fn_gate_ledger.json")["case_count"] == 13
    print(json.dumps({"status": "PASS", "checks": 10}))


if __name__ == "__main__":
    run()
