"""Read-only evidence analysis; hypothetical counts are never model results."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("fix_feasibility.json")
V26 = ROOT / "reports/v26_ai85_optimization"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(tp: int, fp: int, fn: int) -> dict:
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "all_event_gates_gt_085": min(precision, recall, f1) > Fraction(85, 100),
    }


def main() -> None:
    baseline_path = V26 / "experiments/E2/metrics.json"
    evidence_path = V26 / "diagnosis/evidence.json"
    baseline = read_json(baseline_path)
    evidence = read_json(evidence_path)
    entries = evidence["fn_ledger"]["entries"]
    assert len(entries) == baseline["total_gt"] == 20
    assert sum(not row["matched"] for row in entries) == baseline["fn"] == 8
    assert baseline["action_mode"] == "NONE"
    assert (baseline["tp"], baseline["fp"], baseline["fn"]) == (12, 13, 8)
    computed = metrics(12, 13, 8)
    for key in ("precision", "recall", "f1"):
        assert abs(computed[key] - baseline[key]) < 1e-12

    source_files = [
        baseline_path, evidence_path, Path(__file__),
        ROOT / "training/dmd/run_autonomous_v21.py",
        ROOT / "training/dmd/run_verified_v24.py",
        ROOT / "training/dmd/action_adapter.py",
        ROOT / "training/dmd/action_checkpoint.py",
        ROOT / "training/dmd/research_permissions.py",
        ROOT / "backend/ai/events.py",
    ]
    timing_audit = []
    for run in ("E4", "E5"):
        trace_path = V26 / f"experiments/{run}/raw_predictions.jsonl"
        source_files.append(trace_path)
        valid_count = 0
        different_target_count = 0
        admitted_different_target_count = 0
        deltas = []
        examples = []
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                action = row.get("action") or {}
                if not action.get("score_valid") or "score_timestamp" not in action:
                    continue
                valid_count += 1
                obs = row["observation"]
                delta = obs["timestamp"] - action["score_timestamp"]
                if abs(delta) <= 0.001:
                    continue
                different_target_count += 1
                deltas.append(delta)
                if obs.get("fusion_score", 0) > 0:
                    admitted_different_target_count += 1
                    if len(examples) < 2:
                        examples.append({
                            "subject_id": row["subject_id"], "frame_idx": row["frame_idx"],
                            "observation_timestamp": obs["timestamp"],
                            "score_timestamp": action["score_timestamp"],
                            "available_at_timestamp": action["available_at_timestamp"],
                            "delta_seconds": delta, "phone_use": action["phone_use"],
                            "fusion_score": obs["fusion_score"],
                        })
        timing_audit.append({
            "run": run, "valid_centered_scores": valid_count,
            "score_target_differs_from_observation": different_target_count,
            "positive_fusion_with_different_target": admitted_different_target_count,
            "delta_seconds_min": min(deltas) if deltas else None,
            "delta_seconds_max": max(deltas) if deltas else None,
            "examples": examples,
            "interpretation": "Observed timestamp mismatch. Metric impact requires an aligned replay; these counts are not proof of how many events were harmed.",
        })
    snapshots = [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in source_files
    ]
    sessions = {row["subject_id"]: row for row in baseline["sessions"]}
    temporal_support = []
    for event in entries:
        if event["matched"] or event["duration_seconds"] >= 1:
            continue
        session = sessions[event["subject_id"]]
        fps = session["fps"]
        variants = []
        for stride in (15, 10, 5, 3):
            timestamps = [
                frame / fps for frame in range(0, session["source_frames"], stride)
                if event["start_seconds"] <= frame / fps <= event["end_seconds"]
            ]
            count = len(timestamps)
            span = timestamps[-1] - timestamps[0] if count else 0.0
            if stride == session["sample_stride"]:
                assert count == event["sampled_frames"], (event["event_id"], count)
            variants.append({
                "stride": stride, "sample_period_seconds": stride / fps,
                "sampled_frames_inside_gt": count, "sample_span_inside_gt_seconds": span,
                "two_observations_with_050_second_span": count >= 2 and span >= 0.5,
                "two_observations_with_020_second_span": count >= 2 and span >= 0.2,
            })
        temporal_support.append({
            "event_id": event["event_id"], "duration_seconds": event["duration_seconds"],
            "current_primary_root_cause": event["primary_root_cause"], "variants": variants,
        })

    scenarios = [
        ("REMOVE_SIX_DUPLICATE_FP_ONLY", 12, 7, 8),
        ("REMOVE_ALL_FP_ONLY", 12, 0, 8),
        ("RECOVER_ALL_FN_ONLY", 20, 13, 0),
        ("RECOVER_THREE_FN_AND_REMOVE_SIX_FP", 15, 7, 5),
        ("MINIMUM_TP_WITH_LARGEST_PERMITTED_FP", 18, 3, 2),
        ("F1_ALONE_IS_NOT_ALL_GATES", 15, 0, 5),
    ]
    feasible = []
    for tp in range(21):
        allowed = [fp for fp in range(51) if metrics(tp, fp, 20 - tp)["all_event_gates_gt_085"]]
        if allowed:
            feasible.append({"tp": tp, "fn": 20 - tp, "max_fp": max(allowed)})
    assert feasible == [{"tp": tp, "fn": 20 - tp, "max_fp": 3} for tp in (18, 19, 20)]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RESEARCH_DIAGNOSTIC_NO_NEW_INFERENCE_NO_TRAINING",
        "source_snapshots": snapshots,
        "baseline": {"run": "E2", "action_mode": baseline["action_mode"], **computed},
        "fn_root_cause_counts": dict(Counter(row["primary_root_cause"] for row in entries if not row["matched"])),
        "cached_action_timing_audit": timing_audit,
        "hypothetical_count_scenarios": [
            {"name": name, "status": "HYPOTHETICAL_NOT_MEASURED", **metrics(tp, fp, fn)}
            for name, tp, fp, fn in scenarios
        ],
        "feasible_counts_for_all_event_gates_gt_085": feasible,
        "temporal_support_note": (
            "Analytic grid from source FPS and GT intervals, starting at frame 0; no detector/pose/action was run. "
            "Inside-interval sample support is not event recall and does not model outside evidence or matching."
        ),
        "short_fn_sampling_support": temporal_support,
        "four_stage_conditional_retention_example": {
            "each_stage_retention": 0.85, "combined_retention": 0.85 ** 4,
            "equal_retention_for_combined_085": 0.85 ** 0.25,
            "note": "Chain of conditional survival rates for the same GT population; not standalone AP or classification accuracy.",
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.relative_to(ROOT).as_posix(), "baseline": computed,
                      "feasible_counts": feasible, "short_fn_count": len(temporal_support)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
