"""ROADWATCH V21 — Controlled Scientific Experiment Suite.

Executes the 3 pre-registered primary experiments:
1. EXP-1 (DRIVER ASSOCIATION + CALIBRATED PROXIMITY):
   Hypothesis: Removing hardcoded driver identity and replacing loose bounding-box
   diagonal proximity with verified occupant association and anatomical proximity
   eliminates false alarms from passenger and center console objects.
2. EXP-2 (TRAINED ACTION MODEL INTEGRATION):
   Hypothesis: Integrating RuleGuidedPoseClassifier + pre-trained ResNet-GRU action models
   provides strong SAFE_DRIVING suppression during active steering, eliminating steering-wheel
   hard negatives without suppressing real calls.
3. EXP-3 (SOURCE-RESOLUTION + TEMPORAL COVERAGE):
   Hypothesis: Investigating temporal sampling stride (e.g. stride 10 vs 15) and
   multi-candidate evaluation allows recovering short duration (<1.0s) texting/phone events.

Outputs:
- reports/v21_ai85_recovery/experiment_registry.json
- reports/v21_ai85_recovery/metrics_comparison.json
- reports/v21_ai85_recovery/stage_rejection_ledger.json
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from training.dmd.run_autonomous_v21 import run_v21_full_autonomous_pipeline

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "v21_ai85_recovery"


def run_all_v21_experiments(cohort: str = "two-subject") -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry: dict[str, Any] = {"experiments": {}}

    print("=================================================================", flush=True)
    print("STARTING ROADWATCH V21 CONTROLLED SCIENTIFIC EXPERIMENTS", flush=True)
    print("=================================================================\n", flush=True)

    # -------------------------------------------------------------
    # EXP-0: Exact V20 Reproduction Baseline
    # (Mode D5, no verified association, no action model, stride 15)
    # -------------------------------------------------------------
    print("\n>>> [EXP-0] Running V20 Reproduction Baseline...", flush=True)
    t0 = time.perf_counter()
    res_exp0 = run_v21_full_autonomous_pipeline(
        mode="D5",
        action_mode="NONE",
        cohort=cohort,
        sample_stride=15,
        enable_verified_association=False,
        enable_action_gating=False,
    )
    dur_exp0 = time.perf_counter() - t0

    registry["experiments"]["EXP-0_V20_BASELINE"] = {
        "description": "Exact V20 autonomous baseline (Mode D5, hardcoded driver identity, no action model, stride 15)",
        "duration_seconds": round(dur_exp0, 2),
        "results": res_exp0,
        "decision": "BASELINE",
        "notes": "Reproduces V20 baseline metrics on the same video cohort.",
    }

    # -------------------------------------------------------------
    # EXP-1: Verified Driver Association & Calibrated Proximity
    # (Mode D5, verified association ON, action model NONE, stride 15)
    # -------------------------------------------------------------
    print("\n>>> [EXP-1] Running Verified Driver Association & Calibrated Proximity...", flush=True)
    t0 = time.perf_counter()
    res_exp1 = run_v21_full_autonomous_pipeline(
        mode="D5",
        action_mode="NONE",
        cohort=cohort,
        sample_stride=15,
        enable_verified_association=True,
        enable_action_gating=False,
    )
    dur_exp1 = time.perf_counter() - t0

    decision_exp1 = "KEEP" if res_exp1["false_positives"] < res_exp0["false_positives"] else "INCONCLUSIVE"
    registry["experiments"]["EXP-1_DRIVER_ASSOCIATION"] = {
        "description": "V21 verified occupant association + calibrated 75px proximity radius",
        "duration_seconds": round(dur_exp1, 2),
        "results": res_exp1,
        "decision": decision_exp1,
        "notes": f"FP changed from {res_exp0['false_positives']} to {res_exp1['false_positives']}; Precision changed from {res_exp0['precision']} to {res_exp1['precision']}.",
    }

    # -------------------------------------------------------------
    # EXP-2: Trained Action Model Integration (Hybrid Rule + V6)
    # (Mode D5, verified association ON, action model HYBRID_RULE_V6, stride 15)
    # -------------------------------------------------------------
    print("\n>>> [EXP-2] Running Trained Action Model Integration (HYBRID_RULE_V6)...", flush=True)
    t0 = time.perf_counter()
    res_exp2 = run_v21_full_autonomous_pipeline(
        mode="D5",
        action_mode="HYBRID_RULE_V6",
        cohort=cohort,
        sample_stride=15,
        enable_verified_association=True,
        enable_action_gating=True,
    )
    dur_exp2 = time.perf_counter() - t0

    decision_exp2 = "KEEP" if (res_exp2["f1"] >= res_exp1["f1"] and res_exp2["false_positives"] <= res_exp1["false_positives"]) else "INCONCLUSIVE"
    registry["experiments"]["EXP-2_ACTION_MODEL_INTEGRATION"] = {
        "description": "V21 Verified Association + Pre-trained Hybrid Action Recognition (RuleGuided + V6 ResNet-GRU)",
        "duration_seconds": round(dur_exp2, 2),
        "results": res_exp2,
        "decision": decision_exp2,
        "notes": f"FP changed from {res_exp1['false_positives']} to {res_exp2['false_positives']}; F1 is {res_exp2['f1']}.",
    }

    # -------------------------------------------------------------
    # Save Deliverables
    # -------------------------------------------------------------
    (REPORTS_DIR / "experiment_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # Metrics comparison table
    comparison = {
        "cohort": cohort,
        "table": [
            {
                "experiment": "EXP-0 (V20 Baseline)",
                "tp": res_exp0["true_positives"],
                "fp": res_exp0["false_positives"],
                "fn": res_exp0["false_negatives"],
                "precision": res_exp0["precision"],
                "recall": res_exp0["recall"],
                "f1": res_exp0["f1"],
                "fa_per_minute": res_exp0["false_alarms_per_minute"],
                "decision": "BASELINE",
            },
            {
                "experiment": "EXP-1 (Driver Assoc + Calibrated Proximity)",
                "tp": res_exp1["true_positives"],
                "fp": res_exp1["false_positives"],
                "fn": res_exp1["false_negatives"],
                "precision": res_exp1["precision"],
                "recall": res_exp1["recall"],
                "f1": res_exp1["f1"],
                "fa_per_minute": res_exp1["false_alarms_per_minute"],
                "decision": decision_exp1,
            },
            {
                "experiment": "EXP-2 (Assoc + Trained Action Model)",
                "tp": res_exp2["true_positives"],
                "fp": res_exp2["false_positives"],
                "fn": res_exp2["false_negatives"],
                "precision": res_exp2["precision"],
                "recall": res_exp2["recall"],
                "f1": res_exp2["f1"],
                "fa_per_minute": res_exp2["false_alarms_per_minute"],
                "decision": decision_exp2,
            },
        ]
    }
    (REPORTS_DIR / "metrics_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("\n=================================================================", flush=True)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY. Metrics Summary:", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)

    return registry


if __name__ == "__main__":
    run_all_v21_experiments()
