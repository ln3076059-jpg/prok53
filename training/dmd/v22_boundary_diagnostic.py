"""One-factor temporal boundary replay on the exact B0 autonomous observations.

Keeps legacy B0 role assumptions: this is not a verified-driver candidate.
The existing event engine decides activation/release. The reporter retains the
last physically positive timestamp during activation instead of reporting only
the initial activation window. No ground truth enters the boundary tracker.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_checkpoint import sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.run_autonomous_v20 import CONFIG_PATH, POLICY_PATH, ROOT
from training.dmd.run_autonomous_v21 import evaluate_predictions_stage2
from training.dmd.run_verified_v22 import OUTPUT, deduplicate, read_json, write_json
from training.dmd.temporal_eval import TemporalMatchingPolicy
from training.dmd.v22_evaluation import boundary_metrics


class ActiveBoundaryReporter:
    """Observational interval output around the existing activation engine."""

    def __init__(self, config: dict):
        kwargs = {
            k: v
            for k, v in config.items()
            if k in inspect.signature(TemporalEventEngine).parameters
        }
        self.engine = TemporalEventEngine(**kwargs)
        self.active: dict[tuple, dict] = {}
        self.intervals: list[dict] = []

    def add(self, observation: Observation) -> None:
        emitted = self.engine.add(observation)
        for event in emitted:
            if event.event_type != "PHONE":
                continue
            interval = {
                "event_type": "PHONE",
                "start_seconds": event.start_timestamp,
                "end_seconds": event.end_timestamp,
                "confidence": event.confidence,
                "emitted_at_seconds": observation.timestamp,
            }
            self.intervals.append(interval)
            self.active[self.engine._candidate_key(event)] = interval
        for key in list(self.active):
            if key not in self.engine.active_events:
                del self.active[key]
                continue
            observation_key = (
                "PHONE",
                observation.vehicle_context_id,
                observation.track_id if not observation.occupant_id else None,
                observation.occupant_id or observation.occupant_role,
            )
            if (
                key == observation_key
                and observation.class_name == "phone"
                and observation.occupant_role == "driver"
                and observation.phone_context in {"HANDHELD", "HANDHELD_USE", "HANDHELD_PHONE_USE"}
                and observation.confidence >= self.engine.candidate_threshold
            ):
                self.active[key]["end_seconds"] = max(
                    self.active[key]["end_seconds"], observation.timestamp
                )

    def finish(self, cooldown: float) -> list[dict]:
        # No synthetic positive observation at EOF, no extension to unseen timestamps.
        self.active.clear()
        return deduplicate(self.intervals, cooldown)


def run(output: Path = OUTPUT) -> dict:
    config = read_json(CONFIG_PATH)
    cohort = read_json(output / "cohort_manifest.json")
    original = read_json(output / "full_experiments.json")["B0"]
    original_protocol = read_json(output / "evaluation_protocol.json")
    if original["status"] != "COMPLETE":
        raise ValueError("FULL_B0_REQUIRED_FOR_BOUNDARY_COMPARISON")
    for path, key in (
        (CONFIG_PATH, "config_sha256"),
        (POLICY_PATH, "policy_sha256"),
        (ROOT / "training/dmd/temporal_eval.py", "matcher_sha256"),
    ):
        if sha256_file(path) != original_protocol[key]:
            raise ValueError("BOUNDARY_COMPARISON_PROTOCOL_CHANGED")
    frozen = read_json(POLICY_PATH)
    policy = TemporalMatchingPolicy(
        **{
            k: frozen[k]
            for k in ("iou_threshold", "onset_tolerance_seconds", "offset_tolerance_seconds")
        }
    )
    protocol = {
        "scope": "FULL_COHORT_AUTONOMOUS_OBSERVATION_REPLAY_LEGACY_DRIVER_ROLE",
        "changed_factor": "activation-window output versus last physically positive timestamp while active",
        "baseline_prediction_sha256": original["prediction_sha256"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "matcher_sha256": sha256_file(ROOT / "training/dmd/temporal_eval.py"),
        "policy_sha256": sha256_file(POLICY_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "role_verification": False,
        "model_refitting": False,
        "threshold_tuning": False,
        "full_detector_rerun": False,
    }
    write_json(output / "boundary_experiment_protocol.json", protocol)
    subjects = {}
    started = time.perf_counter()
    for session in cohort["sessions"]:
        reporter = ActiveBoundaryReporter(config)
        path = output / "private" / session["subject_id"] / "trace.jsonl"
        inference_summary = read_json(path.parent / "inference_summary.json")
        if sha256_file(path) != inference_summary["trace_sha256"]:
            raise ValueError("BOUNDARY_REPLAY_TRACE_SHA_MISMATCH")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                reporter.add(Observation(**json.loads(line)["observation"]))
        intervals = reporter.finish(config["cooldown_seconds"])
        # Scoring begins only after the autonomous observation replay is complete.
        scored = evaluate_predictions_stage2(
            intervals,
            Path(session["annotation_path"]),
            session["fps"],
            session["source_frames"],
            session["source_seconds"],
            policy,
        )
        gt = [
            i.to_dict()
            for i in parse_dmd_openlabel(
                Path(session["annotation_path"]),
                fps=session["fps"],
                total_frames=session["source_frames"],
            ).phone_intervals
        ]
        subjects[session["subject_id"]] = {
            **scored,
            "predicted_events": intervals,
            "trace_sha256": sha256_file(path),
            "boundary_quality": boundary_metrics(scored["matched_events"], gt, intervals),
        }
    tp, fp, fn = (
        sum(s[k] for s in subjects.values())
        for k in ("true_positives", "false_positives", "false_negatives")
    )
    baseline_matches = read_json(output / "full_event_matches.json")["B0"]
    recovery = {}
    for subject, scored in subjects.items():

        def identity(event: dict) -> tuple:
            return event["start_seconds"], event["end_seconds"]

        old_fn = {identity(g) for g in baseline_matches[subject]["unmatched_gt"]}
        old_tp = {identity(m["ground_truth"]) for m in baseline_matches[subject]["matched_events"]}
        new_fn = {identity(g) for g in scored["unmatched_gt"]}
        new_tp = {identity(m["ground_truth"]) for m in scored["matched_events"]}
        recovery[subject] = {
            "recovered_old_fn": sorted(old_fn - new_fn),
            "old_fn_remaining": sorted(old_fn & new_fn),
            "lost_old_tp": sorted(old_tp - new_tp),
        }
    minutes = cohort["source_minutes"]
    result = {
        "scope": protocol["scope"],
        "per_subject": subjects,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": tp + fn,
        "evaluated_minutes": minutes,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fa_per_minute": fp / minutes,
        "replay_and_scoring_wall_seconds": time.perf_counter() - started,
        "delta_from_B0": {
            "tp": tp - original["tp"],
            "fp": fp - original["fp"],
            "fn": fn - original["fn"],
        },
        "production_approved": False,
        "association_verified": False,
        "recovered_old_fn_count": sum(len(s["recovered_old_fn"]) for s in recovery.values()),
        "old_fn_remaining_count": sum(len(s["old_fn_remaining"]) for s in recovery.values()),
        "lost_old_tp_count": sum(len(s["lost_old_tp"]) for s in recovery.values()),
        "per_subject_recovery_counts": {
            s: {k: len(v) for k, v in values.items()} for s, values in recovery.items()
        },
    }
    write_json(output / "private/boundary_recovery_event_ids.json", recovery)
    write_json(output / "private/boundary_experiment_matches.json", result)
    summary = {k: v for k, v in result.items() if k != "per_subject"}
    write_json(output / "boundary_experiment_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    run()
