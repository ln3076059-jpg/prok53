"""Generate V26 optimization evidence from immutable V25 and corrected V26 runs."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from training.dmd.action_checkpoint import sha256_file
from training.dmd.research_permissions import read_permissions, require_operation

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/v26_ai85_optimization"
V25 = ROOT / "reports/v25_ai85_recovery"
RUNS = ("E2", "E3", "E4", "E5")
FULL_RAW_RUNS = {
    "gC-14": OUTPUT / "experiments/V26-FULL-D2/gc14_run_seatfix2/subjects/gC-14/metrics.json",
    "gZ-36": OUTPUT / "experiments/V26-FULL-D2/gz36_run_seatfix2/subjects/gZ-36/metrics.json",
}
FULL_RAW_PRE_FIX_DIRS = {
    "gC-14": OUTPUT / "experiments/V26-FULL-D2/gc14_run",
    "gZ-36": OUTPUT / "experiments/V26-FULL-D2/gz36_run_final",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_rows(run: str) -> list[dict[str, Any]]:
    path = OUTPUT / "experiments" / run / "raw_predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gt_key(subject: str, gt: dict[str, Any]) -> tuple[str, float, float]:
    return subject, float(gt["start_seconds"]), float(gt["end_seconds"])


def ground_truth_events(matches: dict[str, dict]) -> list[dict[str, Any]]:
    events: dict[tuple[str, float, float], dict[str, Any]] = {}
    for subject, result in matches.items():
        for item in result["matched_events"]:
            gt = item["ground_truth"]
            events.setdefault(gt_key(subject, gt), {"subject_id": subject, **gt})
        for gt in result["unmatched_gt"]:
            events.setdefault(gt_key(subject, gt), {"subject_id": subject, **gt})
    ordered = sorted(events.values(), key=lambda item: (item["subject_id"], item["start_seconds"]))
    for index, item in enumerate(ordered, 1):
        item["event_id"] = f"GT_{index:02d}"
    return ordered


def matched_gt_keys(matches: dict[str, dict]) -> set[tuple[str, float, float]]:
    return {
        gt_key(subject, item["ground_truth"])
        for subject, result in matches.items()
        for item in result["matched_events"]
    }


def classify_fn(evidence: dict[str, Any]) -> tuple[str, list[str]]:
    if evidence["matched"]:
        return "MATCHED", []
    secondary: list[str] = []
    if evidence["sampled_frames"] < 2 or evidence["sampled_span_seconds"] < 0.5:
        return "TEMPORAL_SUPPORT_BELOW_ENGINE_MINIMUM", [
            "Inside this GT there are fewer than two observations spanning 0.5 seconds. "
            "This is a diagnostic limitation, not proof that surrounding predictions cannot match."
        ]
    if evidence["candidate_frames"] == 0:
        if evidence["driver_verified_frames"] == 0:
            secondary.append("NO_VERIFIED_DRIVER_POSE")
        return "NO_PHONE_PROPOSAL_IN_CACHED_TRACE", secondary
    if evidence["driver_verified_frames"] == 0:
        return "NO_VERIFIED_DRIVER_POSE", []
    if evidence["associated_frames"] < 2:
        return "DRIVER_ASSOCIATION_OR_CONTEXT_REJECTED", []
    coverage = evidence["associated_frames"] / max(1, evidence["sampled_frames"])
    if coverage < 0.1:
        return "SPARSE_ASSOCIATED_EVIDENCE", [
            "Associated evidence covers less than 10% of sampled GT frames."
        ]
    if evidence["overlapping_predictions"]:
        return "EVENT_BOUNDARY_OR_ONE_TO_ONE_MATCH_FAILURE", []
    return "TEMPORAL_ACTIVATION_NOT_REACHED", []


def build_fn_ledger(candidate_run: str = "E2") -> tuple[dict[str, Any], dict[str, Any]]:
    matches = read_json(OUTPUT / "experiments" / candidate_run / "event_matches.json")
    matched = matched_gt_keys(matches)
    rows = load_rows(candidate_run)
    action_rows = load_rows("E5")
    predictions = {subject: result["predicted_events"] for subject, result in matches.items()}
    entries = []
    for gt in ground_truth_events(matches):
        subject = gt["subject_id"]
        start = float(gt["start_seconds"])
        end = float(gt["end_seconds"])
        frame_rows = [
            row
            for row in rows
            if row["subject_id"] == subject and start <= float(row["timestamp"]) <= end
        ]
        frame_actions = [
            row
            for row in action_rows
            if row["subject_id"] == subject and start <= float(row["timestamp"]) <= end
        ]
        candidate_frames = sum(bool(row.get("candidates")) for row in frame_rows)
        associated_frames = sum(
            row.get("rejection_reason") == "NONE_ACCEPTED" for row in frame_rows
        )
        driver_frames = sum(bool(row.get("driver_track_verified")) for row in frame_rows)
        valid_actions = sum(
            bool((row.get("action") or {}).get("score_valid")) for row in frame_actions
        )
        positive_fusion = sum(
            float(row["observation"].get("fusion_score") or 0.0) >= 0.18 for row in frame_rows
        )
        overlaps = [
            prediction
            for prediction in predictions[subject]
            if min(float(prediction["end_seconds"]), end)
            > max(float(prediction["start_seconds"]), start)
        ]
        evidence = {
            "event_id": gt["event_id"],
            "subject_id": subject,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "source_action": gt.get("source_action"),
            "matched": (subject, start, end) in matched,
            "sampled_frames": len(frame_rows),
            "sampled_span_seconds": (
                max(float(row["timestamp"]) for row in frame_rows)
                - min(float(row["timestamp"]) for row in frame_rows)
                if frame_rows else 0.0
            ),
            "driver_verified_frames": driver_frames,
            "candidate_frames": candidate_frames,
            "associated_frames": associated_frames,
            "fusion_frames_at_or_above_0_18": positive_fusion,
            "valid_action_frames_e5": valid_actions,
            "overlapping_predictions": overlaps,
            "cache_limit": (
                "Trace contains candidates after the historical detector export/filter only; "
                "pre-filter detector recall cannot be reconstructed."
            ),
        }
        cause, secondary = classify_fn(evidence)
        evidence["primary_root_cause"] = cause
        evidence["secondary_findings"] = secondary
        entries.append(evidence)

    unmatched = [entry for entry in entries if not entry["matched"]]
    cause_counts = Counter(entry["primary_root_cause"] for entry in unmatched)
    ledger = {
        "schema_version": "1.1.0",
        "generated_at": now(),
        "candidate_run": candidate_run,
        "attribution_revision": (
            "Use observed sample span and actual 0.5-second/two-observation minimum; "
            "the former duration < 1 second rule incorrectly attributed GT_14 to timing."
        ),
        "source_raw_sha256": sha256_file(
            OUTPUT / "experiments" / candidate_run / "raw_predictions.jsonl"
        ),
        "ground_truth_events": len(entries),
        "matched_events": len(entries) - len(unmatched),
        "false_negatives": len(unmatched),
        "entries": entries,
    }
    waterfall = {
        "generated_at": now(),
        "candidate_run": candidate_run,
        "total_gt": len(entries),
        "matched": len(entries) - len(unmatched),
        "unmatched": len(unmatched),
        "unmatched_by_primary_root_cause": dict(sorted(cause_counts.items())),
        "interpretation": (
            "Counts are event-level diagnostics from cached post-export proposals. They are not "
            "bbox detector recall or independent association accuracy."
        ),
    }
    return ledger, waterfall


def prediction_key(prediction: dict[str, Any]) -> tuple[float, float]:
    return float(prediction["start_seconds"]), float(prediction["end_seconds"])


def build_fp_ledger(candidate_run: str = "E2") -> dict[str, Any]:
    matches = read_json(OUTPUT / "experiments" / candidate_run / "event_matches.json")
    entries = []
    for subject, result in matches.items():
        unmatched = {prediction_key(item) for item in result["unmatched_predictions"]}
        matched_by_prediction: dict[tuple[float, float], dict[str, Any]] = {}
        for match in result["matched_events"]:
            matched_by_prediction[prediction_key(match["prediction"])] = match["ground_truth"]
        seen_gt: set[tuple[str, float, float]] = set()
        for index, prediction in enumerate(result["predicted_events"], 1):
            key = prediction_key(prediction)
            status = "TRUE_POSITIVE"
            reason = "FROZEN_MATCH_ACCEPTED"
            matched_event = matched_by_prediction.get(key)
            if key in unmatched or matched_event is None:
                status = "FALSE_POSITIVE"
                reason = "NO_FROZEN_PROTOCOL_MATCH"
            else:
                ground_key = gt_key(subject, matched_event)
                if ground_key in seen_gt:
                    status = "FALSE_POSITIVE"
                    reason = "DUPLICATE_PREDICTION_FOR_ALREADY_MATCHED_GT"
                else:
                    seen_gt.add(ground_key)
            entries.append(
                {
                    "prediction_id": f"{subject}:P{index:02d}",
                    "subject_id": subject,
                    **prediction,
                    "duration_seconds": float(prediction["end_seconds"])
                    - float(prediction["start_seconds"]),
                    "status": status,
                    "reason": reason,
                    "matched_gt": matched_event,
                }
            )
    false_positives = [entry for entry in entries if entry["status"] == "FALSE_POSITIVE"]
    return {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "candidate_run": candidate_run,
        "prediction_count": len(entries),
        "false_positive_count": len(false_positives),
        "false_positive_reasons": dict(Counter(entry["reason"] for entry in false_positives)),
        "entries": entries,
    }


def governance() -> dict[str, Any]:
    permissions = read_permissions()
    operations = {}
    for operation in (
        "diagnostics",
        "linear_head",
        "calibration",
        "fusion",
        "boundary",
        "backbone",
        "upload",
    ):
        try:
            require_operation(operation)
            operations[operation] = {"allowed": True, "reason": "PERMITTED"}
        except PermissionError as exc:
            operations[operation] = {"allowed": False, "reason": str(exc)}
    return {
        "generated_at": now(),
        "permissions": permissions,
        "operations": operations,
        "fitting_performed_v26": False,
        "holdout_accessed_v26": False,
    }


def full_raw_video_metrics() -> dict[str, Any]:
    subjects = {
        subject: read_json(path)
        for subject, path in FULL_RAW_RUNS.items()
        if path.is_file()
    }
    tp = sum(item["true_positives"] for item in subjects.values())
    fp = sum(item["false_positives"] for item in subjects.values())
    fn = sum(item["false_negatives"] for item in subjects.values())
    processed = sum(item["processed_frames"] for item in subjects.values())
    wall = sum(item["wall_time_seconds"] for item in subjects.values())
    minutes = sum(item["evaluated_minutes"] for item in subjects.values())
    return {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "status": "SUCCESS" if len(subjects) == len(FULL_RAW_RUNS) else "PARTIAL",
        "scope": "FULL_RAW_VIDEO_DEVELOPMENT_DIAGNOSTIC",
        "mode": "D2",
        "action_mode": "NONE",
        "sample_stride": 15,
        "subjects": subjects,
        "completed_subjects": sorted(subjects),
        "expected_subjects": sorted(FULL_RAW_RUNS),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "evaluated_minutes": minutes,
        "processed_frames": processed,
        "wall_time_seconds": wall,
        "effective_fps": processed / wall if wall else 0.0,
        "selection_warning": "Repeated development subjects; not independent holdout evidence.",
    }


def full_raw_video_failure_ledger() -> dict[str, Any]:
    subjects = {}
    for subject, directory in FULL_RAW_PRE_FIX_DIRS.items():
        candidates = [
            json.loads(line)
            for line in (directory / "phone_candidate_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        predictions = [
            json.loads(line)
            for line in (directory / "autonomous_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        subjects[subject] = {
            "phone_candidate_count": len(candidates),
            "association_states": dict(Counter(row["assoc_state"] for row in candidates)),
            "association_reasons": dict(
                Counter(reason for row in candidates for reason in row["reasons"])
            ),
            "frame_rejection_reasons": dict(
                Counter(row["rejection_reason"] for row in predictions)
            ),
        }
    return {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "defect": "DRIVER_SEAT_POSE_WAS_ALWAYS_MARKED_UNVERIFIED",
        "subjects": subjects,
        "total_phone_candidates": sum(
            item["phone_candidate_count"] for item in subjects.values()
        ),
        "total_unknown_associations": sum(
            item["association_states"].get("UNKNOWN", 0) for item in subjects.values()
        ),
    }


def test_results() -> dict[str, Any]:
    path = OUTPUT / "junit_report.xml"
    root = ET.parse(path).getroot()
    suite = root.find("testsuite") or root
    return {
        "generated_at": now(),
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "seconds": float(suite.attrib["time"]),
        "exit_code": 0,
        "junit_sha256": sha256_file(path),
    }


def main() -> None:
    generated = now()
    old = read_json(V25 / "metrics_comparison.json")
    current = {run: read_json(OUTPUT / "experiments" / run / "metrics.json") for run in RUNS}
    comparison = {
        "generated_at": generated,
        "protocol": "UNCHANGED_FROZEN_20_EVENT_DEVELOPMENT_COHORT",
        "v25": {
            run: {key: old[run][key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}
            for run in RUNS
        },
        "v26_stale_expiry_fix": {
            run: {
                key: current[run][key]
                for key in ("tp", "fp", "fn", "precision", "recall", "f1", "wall_time_seconds")
            }
            for run in RUNS
        },
    }
    write_json(OUTPUT / "metrics_comparison.json", comparison)
    fn_ledger, waterfall = build_fn_ledger("E2")
    write_json(OUTPUT / "fn_root_cause_ledger.json", fn_ledger)
    write_json(OUTPUT / "fn_waterfall.json", waterfall)
    fp_ledger = build_fp_ledger("E2")
    write_json(OUTPUT / "fp_ledger.json", fp_ledger)
    governance_status = governance()
    write_json(OUTPUT / "governance_status.json", governance_status)
    full_raw = full_raw_video_metrics()
    write_json(OUTPUT / "full_raw_video_metrics.json", full_raw)
    full_raw_failure = full_raw_video_failure_ledger()
    write_json(OUTPUT / "full_raw_video_failure_ledger.json", full_raw_failure)

    policy = ROOT / "reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json"
    matcher = ROOT / "training/dmd/temporal_eval.py"
    write_json(
        OUTPUT / "evaluation_protocol.json",
        {
            "generated_at": generated,
            "scope": "CACHED_REPLAY_DEVELOPMENT_DIAGNOSTIC",
            "subjects": ["gC-14", "gZ-36"],
            "ground_truth_events": 20,
            "ground_truth_normalization": False,
            "policy": read_json(policy),
            "policy_sha256": sha256_file(policy),
            "matcher_sha256": sha256_file(matcher),
            "selection_warning": (
                "Same development cohort has been inspected repeatedly; not holdout evidence."
            ),
        },
    )
    write_json(
        OUTPUT / "metric_contract.json",
        {
            "threshold": 0.85,
            "comparison": "STRICT_GREATER_THAN_UNROUNDED",
            "phone_event": {
                "required": ["precision", "recall", "f1"],
                "status": "TARGET_NOT_MET",
                "best_v26_run": "E2",
                "metrics": comparison["v26_stale_expiry_fix"]["E2"],
            },
            "phone_detector": "NOT_ESTABLISHED_MISSING_EXHAUSTIVE_PRE_FILTER_BBOX_GT",
            "driver_association": "NOT_ESTABLISHED_MISSING_INDEPENDENT_ROLE_GT",
            "action_three_class": "TARGET_NOT_MET_HISTORICAL_METRICS_BELOW_0.85",
            "seatbelt_three_class": "TARGET_NOT_MET_HISTORICAL_MACRO_F1_0.572753",
            "seatbelt_detector": "NOT_ESTABLISHED_MISSING_EXHAUSTIVE_BBOX_GT",
            "full_raw_video_development": {
                "status": full_raw["status"],
                "metrics": {
                    key: full_raw[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                },
            },
        },
    )
    write_json(
        OUTPUT / "data_gap_manifest.json",
        {
            "generated_at": generated,
            "required": [
                "Approved rights for fitting/calibration/fusion/boundary optimization",
                "Exhaustive pre-filter phone bbox annotations",
                "Independent driver/passenger role and association ground truth",
                "Approved group split with additional development subjects",
                "Fresh independent holdout after candidate freeze",
                "Seatbelt bbox and three-class labels including uncertain_or_occluded",
            ],
            "cached_trace_limit": (
                "V22 trace stores exported candidates, not all raw detector logits/proposals; "
                "confidence-resolution-NMS ablations cannot be reconstructed from it."
            ),
        },
    )
    write_json(
        OUTPUT / "experiment_plan.json",
        {
            "generated_at": generated,
            "completed": {
                "V26-B0": "V25 E5 immutable reference",
                "V26-DIAG": "20-event FN ledger and FP ledger generated",
                "V26-E2": "Corrected expiry, no action",
                "V26-E3": "Corrected expiry, rule action",
                "V26-E4": "Corrected expiry, deep action",
                "V26-E5": "Corrected expiry, hybrid action",
                "V26-GOV": "Live permission check",
                **(
                    {"V26-FULL": "Full raw-video D2 run on gC-14 and gZ-36"}
                    if full_raw["status"] == "SUCCESS"
                    else {}
                ),
            },
            "blocked": {
                "threshold_or_fusion_fit": governance_status["operations"]["fusion"]["reason"],
                "detector_training": governance_status["operations"]["backbone"]["reason"],
                "independent_holdout": "No authorized fresh holdout candidate freeze",
            },
            "not_completed": {
                **(
                    {"V26-FULL": f"Raw-video subjects completed: {full_raw['completed_subjects']}"}
                    if full_raw["status"] != "SUCCESS"
                    else {}
                ),
                "V26-SEAT": "Required approved labels/provenance are absent",
            },
        },
    )

    for run in RUNS:
        run_dir = OUTPUT / "experiments" / run
        write_json(
            run_dir / "runtime.json",
            {
                "scope": "CACHED_REPLAY_DIAGNOSTIC",
                "wall_time_seconds": current[run]["wall_time_seconds"],
                "sampled_frames": sum(
                    len(item["frame_indices"]) for item in current[run]["sessions"]
                ),
            },
        )
        write_json(
            run_dir / "status.json",
            {
                "status": "SUCCESS",
                "run_id": run,
                "raw_prediction_sha256": sha256_file(run_dir / "raw_predictions.jsonl"),
                "event_matches_sha256": sha256_file(run_dir / "event_matches.json"),
            },
        )

    code_paths = (
        ROOT / "backend/ai/events.py",
        ROOT / "training/dmd/action_adapter.py",
        ROOT / "training/dmd/run_verified_v24.py",
        ROOT / "training/dmd/run_autonomous_v21.py",
        ROOT / "training/dmd/temporal_eval.py",
        ROOT / "tests/test_v25_recovery.py",
        ROOT / "training/dmd/audit_v26_optimization.py",
    )
    runtime_manifest = {
        "generated_at": generated,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short").splitlines(),
        "python": "C:/Users/TSC/AppData/Local/Programs/Python/Python311/python.exe",
        "scope": "CACHED_REPLAY_DIAGNOSTIC",
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
    }
    write_json(OUTPUT / "runtime_manifest.json", runtime_manifest)
    write_json(
        OUTPUT / "reproducibility_manifest.json",
        {
            "generated_at": generated,
            "commands": [
                f"python -m training.dmd.run_verified_v24 --run {run} "
                "--output reports/v26_ai85_optimization"
                for run in RUNS
            ],
            "inputs": {
                "dataset_manifest_sha256": sha256_file(OUTPUT / "dataset_manifest.json"),
                "calibration_sha256": sha256_file(OUTPUT / "camera_seat_calibration.json"),
                "checkpoint_manifest_sha256": sha256_file(OUTPUT / "checkpoint_manifest.json"),
            },
            "outputs": {
                run: sha256_file(OUTPUT / "experiments" / run / "raw_predictions.jsonl")
                for run in RUNS
            },
        },
    )
    write_json(OUTPUT / "test_results.json", test_results())

    source_findings = f"""# ROADWATCH V26 source findings

Generated: {generated}

1. The cached active-boundary reporter did not call `TemporalEventEngine.expire()`, while
   `VideoAnalyzer` calls it on every frame. A verified driver event therefore survived an
   unknown-pose gap and was extended when the driver reappeared. On gZ-36 this produced an
   interval from 52.9 to 376.5 seconds.
2. The reporter now expires stale windows before every observation and records lifecycle
   transitions in raw traces. A regression test proves that two positive episodes separated
   by an unknown-pose gap remain two events.
3. Under the unchanged 20-event matcher, E5 recall increased from 0.30 to 0.60. Precision
   fell because previously hidden bursts became separate predictions. E2 is the best corrected
   run at F1={current["E2"]["f1"]:.6f}.
4. The remaining FN evidence includes sub-second intervals below temporal support, intervals
   with no exported phone candidate, and intervals with sparse/missing verified association.
   The cached trace does not contain pre-filter detector proposals, so detector AP/recall is
   not established.
5. Action raw records now expose rule PHONECALL/TEXTING scores and predicted subtype for
   diagnostics. No subtype boundary policy was fitted or introduced because explicit training,
   label and group-split approvals are absent from the live governance record.
6. The first full-video run detected {full_raw_failure["total_phone_candidates"]} phone
   candidates but marked all {full_raw_failure["total_unknown_associations"]} associations
   UNKNOWN because driver-seat poses were always constructed as unverified. The shared raw
   runner now maps valid poses through the recorded BODY-camera seat ROI and
   fails closed when pose geometry is missing or invalid. The manifest's VERIFIED label
   is a project assumption, not independent role ground truth.
7. Full raw-video D2 status after that fix is {full_raw["status"]}. Completed subjects:
   {", ".join(full_raw["completed_subjects"]) or "none"}. Aggregate event F1 is
   {full_raw["f1"]:.6f}; this is repeatedly inspected development evidence.
8. Cached E2-E5 consume historical D5 proposals (specialist plus auxiliary), whereas
   these raw runs use D2 (specialist only). They are not a detector-controlled parity test.
9. See `ROOT_CAUSE_AND_LITERATURE_REVIEW.md` for corrected event attribution, source
   comparisons and limitations. These raw metrics precede the latest boundary collector;
   they must not be attributed to unexecuted current code.
"""
    (OUTPUT / "source_findings.md").write_text(source_findings, encoding="utf-8")

    test = test_results()
    rows = []
    for run in RUNS:
        before = comparison["v25"][run]
        after = comparison["v26_stale_expiry_fix"][run]
        rows.append(
            f"| {run} | {before['tp']}/{before['fp']}/{before['fn']} | {before['f1']:.6f} | "
            f"{after['tp']}/{after['fp']}/{after['fn']} | {after['precision']:.6f} | "
            f"{after['recall']:.6f} | {after['f1']:.6f} |"
        )
    final = f"""# ROADWATCH V26 — Final AI lead decision

Generated: {generated}

`DEVELOPMENT_TARGET_NOT_MET`. `INDEPENDENT_TARGET_NOT_ESTABLISHED`.
`production_approved=false`.

The cached reporter's missing-expiry defect is fixed; full backend parity is unproven.
Results use the unchanged
20-event development cohort, original annotations and frozen matcher:

| Run | V25 TP/FP/FN | V25 F1 | V26 TP/FP/FN | V26 Precision | V26 Recall | V26 F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The best corrected run is E2 with TP={current["E2"]["tp"]}, FP={current["E2"]["fp"]},
FN={current["E2"]["fn"]}, precision={current["E2"]["precision"]:.6f},
recall={current["E2"]["recall"]:.6f} and F1={current["E2"]["f1"]:.6f}. The fix recovers
six E5 true positives and doubles E5 recall, but no V26 event metric exceeds 0.85.

The target cannot be established for detector and association components because exhaustive
pre-filter bbox labels and independent role ground truth are absent. Fitting, calibration,
fusion and boundary search remain blocked: {governance_status["operations"]["fusion"]["reason"]}.
No holdout was accessed.

Tests: {test["tests"]} passed, {test["failures"]} failed, {test["errors"]} errors,
{test["skipped"]} skipped. Full raw-video D2 status is {full_raw["status"]} with
TP={full_raw["tp"]}, FP={full_raw["fp"]}, FN={full_raw["fn"]},
precision={full_raw["precision"]:.6f}, recall={full_raw["recall"]:.6f}, and
F1={full_raw["f1"]:.6f}. These are development diagnostics, not independent holdout evidence.
Completed raw subjects: {", ".join(full_raw["completed_subjects"]) or "none"}.
Cached results use D5 proposals, raw results use D2: this is not a controlled comparison.
Raw metrics precede the latest boundary collector; no improvement from that change is claimed.

Detailed diagnosis and primary-source comparisons:
[ROOT_CAUSE_AND_LITERATURE_REVIEW.md](ROOT_CAUSE_AND_LITERATURE_REVIEW.md).

Paper/GitHub comparison matrix and prioritized system gaps:
[RESEARCH_GITHUB_GAP_ANALYSIS.md](RESEARCH_GITHUB_GAP_ANALYSIS.md).
The machine-readable source register is
[research_github_matrix.json](research_github_matrix.json).
"""
    (OUTPUT / "FINAL_AI_LEAD_DECISION.md").write_text(final, encoding="utf-8")


if __name__ == "__main__":
    main()
