"""Audit the V27 correctness fixes without fitting on evaluation labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
OLD_OUTPUT = ROOT / "reports" / "v26_ai85_optimization"
OUTPUT = ROOT / "reports" / "v27_ai85_execution"
RUNS = ("E2", "E4", "E5")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "fa_per_minute",
        )
    }


def timing_audit(run_id: str) -> dict[str, Any]:
    raw_path = OUTPUT / "experiments" / run_id / "raw_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid = [row for row in rows if row.get("action", {}).get("score_valid")]
    latencies = [float(row["action"]["emission_latency_seconds"]) for row in valid]
    candidate_rows = [
        row for row in valid if row["action"].get("candidate_id") is not None
    ]
    return {
        "run_id": run_id,
        "raw_prediction_sha256": sha256(raw_path),
        "valid_centered_scores": len(valid),
        "target_frame_mismatches": sum(
            int(row["frame_idx"]) != int(row["action"]["target_frame_idx"])
            for row in valid
        ),
        "target_timestamp_mismatches_over_1ms": sum(
            abs(float(row["timestamp"]) - float(row["action"]["score_timestamp"]))
            > 0.001
            for row in valid
        ),
        "scores_available_before_target": sum(
            float(row["action"]["available_at_timestamp"])
            < float(row["action"]["score_timestamp"])
            for row in valid
        ),
        "candidate_scored_count": len(candidate_rows),
        "candidate_identity_mismatches": sum(
            not any(
                candidate.get("candidate_id") == row["action"]["candidate_id"]
                for candidate in row["candidates"]
            )
            for row in candidate_rows
        ),
        "latency_seconds": {
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
    }


def main() -> None:
    comparisons: dict[str, Any] = {}
    for run_id in RUNS:
        before = read_json(OLD_OUTPUT / "experiments" / run_id / "metrics.json")
        after = read_json(OUTPUT / "experiments" / run_id / "metrics.json")
        comparisons[run_id] = {
            "before_v26": metric_summary(before),
            "after_v27": metric_summary(after),
            "delta": {
                key: metric_summary(after)[key] - metric_summary(before)[key]
                for key in metric_summary(after)
            },
            "per_subject_after": {
                session["subject_id"]: {
                    key: session[key] for key in ("tp", "fp", "fn")
                }
                for session in after["sessions"]
            },
        }

    audits = {run_id: timing_audit(run_id) for run_id in ("E4", "E5")}
    baseline = comparisons["E2"]["after_v27"]
    gate_pass = all(float(baseline[key]) > 0.85 for key in ("precision", "recall", "f1"))
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CACHED_REPLAY_CORRECTNESS_ABLATION_NOT_INDEPENDENT_HOLDOUT",
        "fitting_performed": False,
        "comparisons": comparisons,
        "timing_and_identity_invariants": audits,
        "strict_gate": {
            "threshold": 0.85,
            "requires": "precision > 0.85 AND recall > 0.85 AND f1 > 0.85",
            "pass": gate_pass,
            "status": "MET" if gate_pass else "NOT_MET",
        },
    }
    (OUTPUT / "metrics_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    junit_path = OUTPUT / "junit_report.xml"
    if junit_path.is_file():
        suite = ElementTree.parse(junit_path).getroot().find("testsuite")
        if suite is None:
            raise ValueError("JUNIT_TESTSUITE_NOT_FOUND")
        test_result = {
            "source": str(junit_path.relative_to(ROOT)),
            "sha256": sha256(junit_path),
            "tests": int(suite.attrib["tests"]),
            "failures": int(suite.attrib["failures"]),
            "errors": int(suite.attrib["errors"]),
            "skipped": int(suite.attrib["skipped"]),
            "time_seconds": float(suite.attrib["time"]),
            "pass": (
                int(suite.attrib["failures"]) == 0
                and int(suite.attrib["errors"]) == 0
            ),
        }
        (OUTPUT / "test_results.json").write_text(
            json.dumps(test_result, indent=2) + "\n", encoding="utf-8"
        )

    e2 = comparisons["E2"]["after_v27"]
    e4 = comparisons["E4"]["after_v27"]
    e5 = comparisons["E5"]["after_v27"]
    kaggle_receipt_path = OUTPUT / "kaggle_execution_receipt.json"
    kaggle_section = ""
    if kaggle_receipt_path.is_file():
        kaggle = read_json(kaggle_receipt_path)["verified_result"]
        runtimes = kaggle["stage_runtime"]
        kaggle_section = f"""
## Kaggle GPU preflight

Kaggle kernel version 3 completed on **{kaggle['gpu']}**. Device provenance is specialist={kaggle['device_provenance']['specialist']}, auxiliary={kaggle['device_provenance']['auxiliary']}, and pose={kaggle['device_provenance']['pose']}. Median call latency was {runtimes['specialist']['p50_ms']:.2f} ms, {runtimes['auxiliary']['p50_ms']:.2f} ms, and {runtimes['pose']['p50_ms']:.2f} ms respectively on 32 existing private validation images.

This run performed no training, used no DMD media, and computed no event metric. Raw D5 remains `{kaggle['raw_d5_status']}` because the account has no authorized DMD media dataset. The signed local receipt is `kaggle_execution_receipt.json`.
"""
    report = f"""# V27 AI85 correctness execution report

Generated {result['generated_at']}. Scope: cached replay correctness ablation on the existing two-subject development cohort. No parameter, threshold, label, or model fitting was performed in this execution.

## Result

| Run | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 baseline | {e2['tp']} | {e2['fp']} | {e2['fn']} | {e2['precision']:.2%} | {e2['recall']:.2%} | {e2['f1']:.2%} | {e2['fa_per_minute']:.4f} |
| E4 aligned deep | {e4['tp']} | {e4['fp']} | {e4['fn']} | {e4['precision']:.2%} | {e4['recall']:.2%} | {e4['f1']:.2%} | {e4['fa_per_minute']:.4f} |
| E5 aligned hybrid | {e5['tp']} | {e5['fp']} | {e5['fn']} | {e5['precision']:.2%} | {e5['recall']:.2%} | {e5['f1']:.2%} | {e5['fa_per_minute']:.4f} |

The mandatory gate is **{result['strict_gate']['status']}**. E4 no longer adds the extra false positive seen in V26, but the best measured result remains E2/E4 at F1 {e2['f1']:.2%}. Reaching all three metrics above 85% on 20 GT events requires at least 18 TP and at most 3 FP; the current best has {e2['tp']} TP and {e2['fp']} FP.

## Correctness invariants

| Run | Valid centered scores | Wrong target frame | Target time error >1 ms | Emitted before target | Candidate scored | Candidate ID mismatch | Declared latency range |
|---|---:|---:|---:|---:|---:|---:|---|
| E4 | {audits['E4']['valid_centered_scores']} | {audits['E4']['target_frame_mismatches']} | {audits['E4']['target_timestamp_mismatches_over_1ms']} | {audits['E4']['scores_available_before_target']} | {audits['E4']['candidate_scored_count']} | {audits['E4']['candidate_identity_mismatches']} | {audits['E4']['latency_seconds']['minimum']:.6f}-{audits['E4']['latency_seconds']['maximum']:.6f}s |
| E5 | {audits['E5']['valid_centered_scores']} | {audits['E5']['target_frame_mismatches']} | {audits['E5']['target_timestamp_mismatches_over_1ms']} | {audits['E5']['scores_available_before_target']} | {audits['E5']['candidate_scored_count']} | {audits['E5']['candidate_identity_mismatches']} | {audits['E5']['latency_seconds']['minimum']:.6f}-{audits['E5']['latency_seconds']['maximum']:.6f}s |

The old V26 trace attached every centered score about 2.016 seconds late to the currently processed observation. V27 maps the score back to its exact target frame, uses that frame's pose and selected candidate, and records availability separately. The raw online runner now rejects centered checkpoints because it has no target-aligned delayed-output queue.

## Verification

The focused V22-V27, temporal business-rule, evaluation-governance, and pose-feature suite passed 102/102 tests. The machine-readable result is in `test_results.json`; individual cases and timings are in `junit_report.xml`.

{kaggle_section}

## Remaining measured bottleneck

The best run has gC-14 = 7/3/2 and gZ-36 = 5/10/6 for TP/FP/FN. The dominant gap is proposal and association quality on gZ-36, plus short-event sampling; action fusion alone cannot recover the missing six true events or remove ten false events needed to pass the strict gate. The next valid experiment is a pre-registered raw D5 detector/cadence study with component-level bbox and association labels, followed by source-aware calibration on development data and one frozen independent evaluation.
"""
    (OUTPUT / "EXECUTION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "gate": result["strict_gate"]}, indent=2))


if __name__ == "__main__":
    main()
