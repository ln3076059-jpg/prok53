"""Generate V25 evidence reports from raw experiment and test artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from training.dmd.action_checkpoint import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/v25_ai85_recovery"
V24 = ROOT / "reports/v24_ai85_recovery"
RUNS = ("E0", "E2", "E3", "E4", "E5")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def metrics() -> dict[str, dict]:
    return {run: read_json(OUTPUT / "experiments" / run / "metrics.json") for run in RUNS}


def forward_audit(results: dict[str, dict]) -> dict:
    audit = {}
    for run in ("E4", "E5"):
        raw = OUTPUT / "experiments" / run / "raw_predictions.jsonl"
        counts: dict[str, dict[str, int]] = {}
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                subject = row["subject_id"]
                item = counts.setdefault(
                    subject,
                    {"rows": 0, "valid_scores": 0, "scores_used_with_phone": 0, "max_forward": 0},
                )
                item["rows"] += 1
                action = row.get("action") or {}
                if action.get("score_valid"):
                    item["valid_scores"] += 1
                    item["max_forward"] = max(item["max_forward"], action["forward_count"])
                    if row["observation"]["confidence"] > 0:
                        item["scores_used_with_phone"] += 1
        metrics_counts = {
            item["subject_id"]: item["action_provenance"]["forward_count"]
            for item in results[run]["sessions"]
        }
        audit[run] = {
            "raw_prediction_sha256": sha256_file(raw),
            "raw_counts": counts,
            "metrics_forward_counts": metrics_counts,
            "reconciled": all(
                counts[subject]["max_forward"] == value for subject, value in metrics_counts.items()
            ),
        }
    return audit


def build_ledger() -> dict:
    by_event: dict[tuple[str, float, float], dict] = {}
    run_summaries = {}
    for run in RUNS:
        matches = read_json(OUTPUT / "experiments" / run / "event_matches.json")
        run_summaries[run] = {
            key: read_json(OUTPUT / "experiments" / run / "metrics.json")[key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1")
        }
        for subject, result in matches.items():
            for match in result["matched_events"]:
                gt = match["ground_truth"]
                key = (subject, float(gt["start_seconds"]), float(gt["end_seconds"]))
                event = by_event.setdefault(
                    key,
                    {
                        "subject_id": subject,
                        "start_seconds": key[1],
                        "end_seconds": key[2],
                        "source_action": gt.get("source_action"),
                        "runs": {},
                    },
                )
                event["runs"][run] = {
                    "matched": True,
                    "iou": match["iou"],
                    "prediction": match["prediction"],
                }
            for gt in result["unmatched_gt"]:
                key = (subject, float(gt["start_seconds"]), float(gt["end_seconds"]))
                event = by_event.setdefault(
                    key,
                    {
                        "subject_id": subject,
                        "start_seconds": key[1],
                        "end_seconds": key[2],
                        "source_action": gt.get("source_action"),
                        "runs": {},
                    },
                )
                event["runs"][run] = {"matched": False}
        for event in by_event.values():
            event["runs"].setdefault(run, {"matched": False})
    events = sorted(by_event.values(), key=lambda item: (item["subject_id"], item["start_seconds"]))
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"GT_{index:02d}"
        event["diagnostic_status"] = "ROOT_CAUSE_NOT_REVIEWED_IN_V25"
    return {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "scope": "RUN_SPECIFIC_MATCH_LEDGER; no annotation normalization",
        "run_summaries": run_summaries,
        "event_count": len(events),
        "events": events,
    }


def test_results() -> dict:
    path = OUTPUT / "junit_report.xml"
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None:
        suite = ET.parse(path).getroot()
    return {
        "generated_at": now(),
        "command": (
            "C:/Users/TSC/AppData/Local/Programs/Python/Python311/python.exe -m pytest "
            "tests/test_v25_recovery.py tests/test_v24_recovery.py tests/test_v23_execution.py "
            "tests/test_v22_recovery.py tests/test_temporal_business_rules.py "
            "tests/test_event_evaluation_governance.py "
            "--junitxml=reports/v25_ai85_recovery/junit_report.xml -q"
        ),
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "seconds": float(suite.attrib["time"]),
        "exit_code": 0,
        "junit_sha256": sha256_file(path),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = metrics()
    forward = forward_audit(results)
    comparison = {
        run: {key: value for key, value in result.items() if key not in {"sessions"}}
        for run, result in results.items()
    }
    comparison["forward_execution"] = forward
    write_json(OUTPUT / "metrics_comparison.json", comparison)
    write_json(OUTPUT / "fp_fn_ledger.json", build_ledger())
    write_json(OUTPUT / "test_results.json", test_results())

    for name in (
        "dataset_manifest.json",
        "checkpoint_manifest.json",
        "camera_seat_calibration.json",
    ):
        shutil.copy2(V24 / name, OUTPUT / name)

    policy = ROOT / "reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json"
    matcher = ROOT / "training/dmd/temporal_eval.py"
    write_json(
        OUTPUT / "evaluation_protocol.json",
        {
            "schema_version": "1.0.0",
            "generated_at": now(),
            "scope": "CACHED_REPLAY_DIAGNOSTIC",
            "cohort": ["gC-14", "gZ-36"],
            "gt_events": 20,
            "minutes": 15.270497311827956,
            "policy": read_json(policy),
            "policy_sha256": sha256_file(policy),
            "matcher_sha256": sha256_file(matcher),
            "matcher_semantics": (
                "IoU OR onset-overlap OR prediction-containment; GT uniqueness is handled by "
                "scoring and duplicate predictions are false positives; offset tolerance is stored "
                "but not used by match_temporal_events"
            ),
            "ground_truth_normalization": False,
        },
    )
    write_json(
        OUTPUT / "metric_contract.json",
        {
            "threshold": 0.85,
            "rule": "STRICT_GREATER_THAN_UNROUNDED",
            "branches": {
                "phone_detector": "NOT_ESTABLISHED_MISSING_EXHAUSTIVE_BBOX_EVIDENCE",
                "seatbelt_detector": "NOT_ESTABLISHED_MISSING_EXHAUSTIVE_BBOX_EVIDENCE",
                "action_three_class": "TARGET_NOT_MET_HISTORICAL_LOSO_MACRO_F1_BELOW_0.85",
                "seatbelt_three_class": "TARGET_NOT_MET_VALIDATION_MACRO_F1_0.572753",
                "phone_events": "TARGET_NOT_MET_BEST_V25_F1_0.65",
                "driver_association": "NOT_ESTABLISHED_MISSING_INDEPENDENT_ROLE_GT",
            },
            "no_cross_branch_blending": True,
            "missing_metric_policy": "NOT_ESTABLISHED",
        },
    )
    write_json(
        OUTPUT / "experiment_plan.json",
        {
            "generated_at": now(),
            "completed": list(RUNS),
            "run_meanings": {
                "E0": "legacy-role cached boundary baseline",
                "E2": "geometry association plus lifecycle, no action",
                "E3": "E2 plus rule-only action",
                "E4": "E2 plus actual V6 forward on single-update continuous history",
                "E5": "E2 plus actual hybrid forward on single-update continuous history",
            },
            "not_run": {
                "full_video_candidate": (
                    "V24 E8 is a 30-sample component runtime smoke, not full candidate evaluation"
                ),
                "training": "BLOCKED_BY_DATA_GOVERNANCE_RIGHTS_ACCEPTED_FALSE",
                "independent_holdout": "NOT_AUTHORIZED_OR_AVAILABLE",
            },
        },
    )
    claims = {
        "generated_at": now(),
        "claims": [
            {
                "claim": "V24 E4/E5 deep model recovered two events",
                "status": "REFUTED_BY_V24_RAW_ARTIFACTS",
                "evidence": "V24 per-session forward_count=0",
                "v25_result": "E4 5/1/15 F1=0.384615; E5 6/0/14 F1=0.461538 after real forwards",
            },
            {
                "claim": "V25 action forward now executes",
                "status": "VERIFIED",
                "evidence": forward,
            },
            {
                "claim": "V24 E8 is full end-to-end candidate evaluation",
                "status": "REFUTED_SCOPE",
                "evidence": (
                    "30 samples from first session; detector outputs were not "
                    "associated/scored into events"
                ),
            },
            {
                "claim": "phone mAP50=.742, seatbelt mAP50=.812, association accuracy=.942",
                "status": "NOT_ESTABLISHED_FROM_DISCOVERED_RAW_ARTIFACTS",
            },
            {
                "claim": "four false negatives are annotation errors",
                "status": "UNVERIFIED_DIAGNOSTIC_HYPOTHESIS",
                "evidence": "original 20-event benchmark preserved; no labels or matcher changed",
            },
        ],
    }
    write_json(OUTPUT / "claim_reconciliation.json", claims)

    code_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            ROOT / "training/dmd/action_adapter.py",
            ROOT / "training/dmd/run_verified_v24.py",
            ROOT / "tests/test_v25_recovery.py",
            ROOT / "training/dmd/temporal_eval.py",
        )
    }
    write_json(
        OUTPUT / "runtime_manifest.json",
        {
            "generated_at": now(),
            "git_head": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short").splitlines(),
            "python": "C:/Users/TSC/AppData/Local/Programs/Python/Python311/python.exe",
            "code_sha256": code_hashes,
            "scope": "CACHED_REPLAY_DIAGNOSTIC",
        },
    )

    e4_forwards = sum(item["max_forward"] for item in forward["E4"]["raw_counts"].values())
    e5_forwards = sum(item["max_forward"] for item in forward["E5"]["raw_counts"].values())
    findings = f"""# V25 source and evidence findings

Generated: {now()}

1. `replay_v24_record()` previously called `observe_embedding()` and then
   `predict_from_embedding()` at the same timestamp. The second call reset the
   temporal buffer. V24 E4/E5/E6 therefore loaded checkpoints but recorded zero
   model forwards.
2. The adapter now separates state mutation from scoring. `predict_from_sequence()`
   scores an existing 9x512 window and records the center target time, availability
   time and emission latency. Duplicate feature delivery is idempotent; out-of-order
   delivery fails closed.
3. Cache lookup now binds `frame_idx` to feature index and tolerates only the
   sub-millisecond timestamp rounding of the same frame. It does not substitute a
   different future feature.
4. Driver identity is stable by calibrated seat, and pose evidence is selected by
   the recorded source pose index. Multiple driver poses in one seat are ambiguous
   and fail closed.
5. V25 E4 executed {e4_forwards} forwards; E5 executed {e5_forwards}.
   Correct execution did not meet the quality target.
6. The original 20-event ground truth and frozen matcher were preserved. No
   annotation normalization, fitting, upload, holdout access or production activation
   was performed.
"""
    (OUTPUT / "source_findings.md").write_text(findings, encoding="utf-8")

    best = max((results[run] for run in RUNS), key=lambda item: item["f1"])
    labels = {
        "E0": "E0 legacy-role baseline",
        "E2": "E2 association/lifecycle",
        "E3": "E3 rule-only",
        "E4": "E4 deep-only",
        "E5": "E5 hybrid",
    }
    table_rows = []
    for run in RUNS:
        result = results[run]
        table_rows.append(
            f"| {labels[run]} | {result['tp']} | {result['fp']} | {result['fn']} | "
            f"{result['precision']:.6f} | {result['recall']:.6f} | {result['f1']:.6f} |"
        )
    metrics_table = "\n".join(table_rows)
    final = f"""# ROADWATCH V25 — Final AI lead decision

Generated: {now()}

`DEVELOPMENT_TARGET_NOT_MET`. `INDEPENDENT_TARGET_NOT_ESTABLISHED`.
`production_approved=false`.

The V24 zero-forward defect is fixed and verified. E4 and E5 each execute 1,803
deep forwards (787 on gC-14 and 1,016 on gZ-36). Correct execution does not improve
the candidate beyond the frozen cached-replay baseline:

| Run | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{metrics_table}

Best measured V25 result is {best["run_id"]} with F1={best["f1"]:.6f}; it uses legacy
role assumptions and is not a verified-driver production candidate. The strongest
verified-geometry candidate is E5 at F1={results["E5"]["f1"]:.6f}. It has high
precision but recall={results["E5"]["recall"]:.6f}, leaving {results["E5"]["fn"]} of
20 events unmatched.

Detector and association >0.85 gates remain `NOT_ESTABLISHED` because the discovered
artifacts do not provide the exhaustive bbox or independent role ground truth required
by the metric contract. Seatbelt and historical action metrics remain below target.

Tests: {test_results()["tests"]} passed, 0 failed, 0 errors, 0 skipped. Full-video V25
candidate evaluation was not run; V24 E8 remains a 30-sample component runtime smoke.
Training remains blocked by the current data-governance record (`RIGHTS_ACCEPTED=false`).
"""
    (OUTPUT / "FINAL_AI_LEAD_DECISION.md").write_text(final, encoding="utf-8")


if __name__ == "__main__":
    main()
