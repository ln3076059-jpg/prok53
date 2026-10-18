"""V15 read-only research reconciliation and synthetic sliced-inference diagnostics."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from training.common import sha256_file
from training.dmd.autonomous_phone import PhoneProposal
from training.dmd.behavior_roi import Box
from training.dmd.research_permissions import read_permissions, require_operation
from training.dmd.sliced_phone import SlicedPhoneDetector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v15_research"


def load(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class SyntheticDetector:
    """Deterministic geometry fixture; this is not an accuracy model."""

    model_sha256 = "SYNTHETIC_GEOMETRY_FIXTURE_NO_CHECKPOINT"

    def detect(self, frame: np.ndarray, confidence_floor: float) -> list[PhoneProposal]:
        height, width = frame.shape[:2]
        if min(height, width) < 32:
            return []
        return [
            PhoneProposal(
                Box(width * 0.42, height * 0.42, width * 0.58, height * 0.64),
                max(confidence_floor, 0.75),
                "SYNTHETIC_GEOMETRY_FIXTURE",
                self.model_sha256,
            )
        ]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = load("reports/v11_research/fn_root_cause.json")
    stages = Counter(stage for case in source["cases"] for stage in case["failure_stages"])
    gates = Counter(case["gate_rejection_reason"] for case in source["cases"])
    ledger = {
        "schema_version": 1,
        "evidence_level": "LEVEL_1_DIAGNOSTIC",
        "scope": source["scope"],
        "source": "reports/v11_research/fn_root_cause.json",
        "source_sha256": sha256_file(ROOT / "reports/v11_research/fn_root_cause.json"),
        "case_count": len(source["cases"]),
        "nonexclusive_failure_stage_counts": dict(stages),
        "primary_gate_rejection_counts": dict(gates),
        "input_candidates": 13,
        "admitted_events": 0,
        "rejected_events": 13,
        "rejected_true_phone_use_cases": 13,
        "potential_recovered_tp": None,
        "potential_new_fp": None,
        "limitation": (
            "FN-selected cases have no negative denominator; proposal yield is not event recovery."
        ),
        "cases": [
            {
                "event_id": case["event_id"],
                "subject_id": case["subject_id"],
                "phone_visibility": case["phone_visibility"],
                "specialist_samples_with_box": case["object_detection"][
                    "specialist_samples_with_box"
                ],
                "auxiliary_candidate_count": case["object_detection"]["candidate_count"],
                "driver_association": case["driver_association"]["state"],
                "action_t0_max": case["action_evidence"]["T0_max"],
                "action_t1_dense_max": case["action_evidence"]["T1_dense_max"],
                "gate_rejection_reason": case["gate_rejection_reason"],
                "failure_stages": case["failure_stages"],
            }
            for case in source["cases"]
        ],
    }
    write("fn_gate_ledger.json", ledger)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    adapter = SlicedPhoneDetector(SyntheticDetector(), tile_size=(384, 384), overlap=0.25)
    timings = []
    result = None
    for _ in range(30):
        started = perf_counter()
        result = adapter.detect_roi(frame, Box(0, 72, 832, 684), 0.02)
        timings.append((perf_counter() - started) * 1000)
    assert result is not None
    ordered = sorted(timings)
    sliced = {
        "experiment_id": "V15-A-SLICED-PHONE",
        "evidence_level": "LEVEL_1_SYNTHETIC_ENGINEERING",
        "input_resolution": [1280, 720],
        "roi": [0, 72, 832, 684],
        "tile_size": [384, 384],
        "overlap": 0.25,
        "tile_count": len(result.tiles),
        "raw_proposal_count": result.raw_proposal_count,
        "merged_proposal_count": result.merged_proposal_count,
        "latency_ms": {
            "p50": median(ordered),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "p99": ordered[int(0.99 * (len(ordered) - 1))],
        },
        "model_inference_included": False,
        "accuracy_metrics": None,
        "interpretation": (
            "Measures adapter geometry/overhead only; not detector or full-pipeline latency."
        ),
    }
    write("sliced_phone_diagnostic.json", sliced)

    permissions = read_permissions()
    denied = {}
    for operation in ("linear_head", "calibration", "fusion", "boundary", "backbone", "upload"):
        try:
            require_operation(operation)
        except PermissionError as exc:
            denied[operation] = str(exc)
    v14_data = load("reports/v14_research/data_manifest.json")
    write(
        "dataset_manifest.json",
        {
            "source": "reports/v14_research/data_manifest.json",
            "source_sha256": sha256_file(ROOT / "reports/v14_research/data_manifest.json"),
            "development_subjects": v14_data["development_subjects"],
            "protected_holdouts": ["gZ-37", "gE-28"],
            "protected_holdouts_accessed": False,
            "permissions": {key: permissions.get(key) for key in (
                "RIGHTS_STATUS", "RIGHTS_ACCEPTED", "CANONICAL_ELIGIBLE",
                "TEMPORAL_DEVELOPMENT_ELIGIBLE", "UNTOUCHED_HOLDOUT_ELIGIBLE"
            )},
            "denied_operations": denied,
            "independent_holdout": None,
            "seatbelt_pending_labels": v14_data["seatbelt_review"],
        },
    )
    model = load("reports/v14_research/model_manifest.json")
    model.update(
        sliced_adapter="NO_NEW_WEIGHTS",
        active_model_changed=False,
        production_approved=False,
    )
    write("model_manifest.json", model)
    metrics = load("reports/v14_research/metrics_comparison.json")
    metrics.update(
        v15_new_accuracy_metrics=None,
        v15_sliced_accuracy_metrics=None,
        v15_event_metrics=None,
        independent_ai85_established=False,
        production_approved=False,
    )
    write("metrics_comparison.json", metrics)
    write(
        "confusion_matrices.json",
        {
            "source": "reports/v14_research/metrics_comparison.json",
            "e3_pooled": metrics["clip_selected_e3"]["pooled"]["confusion_matrix"],
            "new_v15_confusion_matrix": None,
        },
    )
    write(
        "event_match_records.json",
        {
            "source": "reports/v14_research/event_match_records.json",
            "source_sha256": sha256_file(ROOT / "reports/v14_research/event_match_records.json"),
            "ground_truth_changed": False,
            "matcher_changed": False,
            "v15_real_event_replay": "NOT_RUN_NO_NEW_ACCURACY_MODEL_OR_VERIFIED_ASSOCIATION",
        },
    )
    write(
        "experiment_registry.json",
        {
            "source_head": head,
            "plan_sha256": sha256_file(OUT / "experiment_plan.json"),
            "experiments": {
                "V15-A-SLICED-PHONE": "KEEP_ENGINEERING_ACCURACY_BLOCKED",
                "V15-B-FN-GATE-LEDGER": "KEEP_DIAGNOSTIC",
                "V15-C-ABSTENTION-INTEGRITY": "KEEP_ENGINEERING",
            },
            "training_performed": False,
            "protected_holdouts_accessed": False,
            "production_approved": False,
        },
    )
    write(
        "environment.json",
        {
            "head_before_v15_changes": head,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": "NOT_QUERIED_BY_MODEL_BENCHMARK",
            "benchmark_scope": "SYNTHETIC_ADAPTER_GEOMETRY_ONLY",
        },
    )


if __name__ == "__main__":
    run()
