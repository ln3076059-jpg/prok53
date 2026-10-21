"""Frozen-cohort recovery runs. Resume completed sessions; retain interrupted attempts.

This runner reuses V20 inference and the existing temporal/association engines.
No annotation object or timestamp list enters inference or evidence replay.
Deep runs fail closed when checkpoint provenance or temporal alignment is missing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from backend.ai.auxiliary import PoseEvidence
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_checkpoint import CheckpointSafetyError, sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import DriverPhoneAssociation
from training.dmd.run_autonomous_v20 import (
    AUXILIARY_PATH,
    CONFIG_PATH,
    POLICY_PATH,
    POSE_PATH,
    ROOT,
    SPECIALIST_PATH,
    AutonomousVideoInferenceEngine,
)
from training.dmd.run_autonomous_v21 import (
    compute_calibrated_proximity,
    evaluate_predictions_stage2,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy
from training.dmd.v22_evaluation import boundary_metrics, validate_full_coverage

OUTPUT = ROOT / "reports/v22_verified_ai85"
SCOPE = "FULL_COHORT_AUTONOMOUS_EXPERIMENT"
MODES = {
    "B0": "NONE",
    "B1": "NONE",
    "B2": "RULE_ONLY",
    "B3": "V6_HIERARCHICAL",
    "B4": "HYBRID_RULE_V6",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cohort() -> dict:
    baseline = read_json(ROOT / "reports/v21_ai85_recovery/dataset_manifest.json")
    subjects = baseline["primary_benchmark_cohort"]
    if subjects != ["gC-14", "gZ-36"]:
        raise ValueError("UNEXPECTED_BASELINE_COHORT")
    sessions = []
    missing = []
    for item in baseline["subjects"]:
        if item["subject_id"] not in subjects:
            continue
        video, annotation = ROOT / item["video_file"], ROOT / item["annotation_file"]
        missing.extend(str(path) for path in (video, annotation) if not path.is_file())
        if missing:
            continue
        video_sha, annotation_sha = sha256_file(video), sha256_file(annotation)
        if video_sha != item["video_sha256"] or annotation_sha != item["ann_sha256"]:
            raise ValueError("BASELINE_SOURCE_SHA_MISMATCH")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"UNREADABLE_VIDEO: {video}")
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        if fps <= 0 or frames <= 0:
            raise ValueError("INVALID_VIDEO_METADATA")
        annotation_data = parse_dmd_openlabel(annotation, fps=fps, total_frames=frames)
        sessions.append(
            {
                "subject_id": item["subject_id"],
                "session_id": video.stem,
                "camera_id": "BODY",
                "video_path": str(video),
                "annotation_path": str(annotation),
                "video_sha256": video_sha,
                "annotation_sha256": annotation_sha,
                "fps": fps,
                "source_frames": frames,
                "width": width,
                "height": height,
                "source_seconds": frames / fps,
                "gt_count": len(annotation_data.phone_intervals),
                "role_calibration_status": "MISSING_VERIFIED_SEAT_CALIBRATION",
            }
        )
    if missing:
        return {"status": "BLOCKED_MISSING_DATA", "missing": missing, "sessions": sessions}
    if len(sessions) != len(subjects) or sum(s["gt_count"] for s in sessions) != 20:
        raise ValueError("BASELINE_GT_OR_SUBJECT_COUNT_MISMATCH")
    return {
        "status": "VERIFIED_LOCAL_FILES",
        "subjects": subjects,
        "sessions": sessions,
        "total_gt_events": sum(s["gt_count"] for s in sessions),
        "source_minutes": sum(s["source_seconds"] for s in sessions) / 60,
        "scope": "DEVELOPMENT_ONLY",
        "fitting_performed": False,
    }


def replay_verified_record(
    record: dict, adapter: ActionInferenceAdapter | None
) -> tuple[Observation, dict]:
    """Candidate geometry alone never verifies driver identity or mounted context."""
    pose = PoseEvidence(**record["poses"][0]) if record["poses"] else None
    candidates = []
    for detection in record["candidates"]:
        local = detection["local_box"]
        hand = compute_calibrated_proximity(pose.hand_points, local, 100) if pose else 0.0
        face = compute_calibrated_proximity(pose.face_points, local, 85) if pose else 0.0
        context = "HANDHELD_USE" if max(hand, face) >= 0.35 else "UNKNOWN"
        association = DriverPhoneAssociation(minimum_containment=0.35).associate(
            Box(*local),
            (),
            phone_context=context,
            phone_confidence=detection["confidence"],
            hand_proximity=hand,
            face_proximity=face,
            context_verified=False,
        )
        candidates.append(
            {
                **detection,
                "hand_proximity": hand,
                "face_proximity": face,
                "association": association.as_dict(),
                "context": context,
            }
        )
    action = None
    if adapter:
        # RULE_ONLY does not consume image pixels; no cached labels enter its interface.
        action = adapter.predict(
            driver_crop=None,
            pose=pose,
            phone_conf=max((d["confidence"] for d in candidates), default=0.0),
            phone_box_local=candidates[0]["local_box"] if candidates else None,
        ).to_dict()
    observation = Observation(
        timestamp=record["observation"]["timestamp"],
        class_name="phone",
        confidence=0.0,
        track_id=None,
        occupant_role="UNKNOWN",
        vehicle_context_id="DMD_CABIN",
        phone_context="UNKNOWN",
        fusion_score=0.0,
        evidence_source="V22_UNVERIFIED_ROLE",
    )
    return observation, {
        "candidates": candidates,
        "action": action,
        "rejection_reason": "UNVERIFIED_OCCUPANT_ASSOCIATION"
        if candidates
        else "NO_PHONE_DETECTED",
        "role_calibration_status": "MISSING_VERIFIED_SEAT_CALIBRATION",
    }


def deduplicate(events: list[dict], cooldown: float) -> list[dict]:
    result = []
    for event in events:
        if result and event["start_seconds"] <= result[-1]["end_seconds"] + cooldown:
            result[-1]["end_seconds"] = max(result[-1]["end_seconds"], event["end_seconds"])
        else:
            result.append(dict(event))
    return result


def extract_session(
    video_path: Path, folder: Path, config: dict, stride: int, limit: int | None
) -> dict:
    """Inference-only boundary: receives video and model config, never annotations."""
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "inference_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary["trace_sha256"] != sha256_file(folder / "trace.jsonl"):
            raise ValueError("RESUME_TRACE_SHA_MISMATCH")
        return summary
    attempt = time.time_ns()
    partial = folder / f"trace_attempt_{attempt}.partial.jsonl"
    started = time.perf_counter()
    count = 0
    try:
        with partial.open("x", encoding="utf-8") as handle:

            def sink(record: dict) -> None:
                nonlocal count
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                count += 1
                if count % 100 == 0:
                    write_json(
                        folder / "progress.json",
                        {
                            "status": "RUNNING",
                            "sampled_frames": count,
                            "last_frame_index": record["frame_idx"],
                            "elapsed_seconds": time.perf_counter() - started,
                            "resume_policy": "COMPLETED_SESSIONS_ONLY; interrupted attempt preserved",
                        },
                    )

            inference = AutonomousVideoInferenceEngine(sample_stride=stride).run_session_inference(
                video_path,
                config,
                max_frames=limit,
                record_sink=sink,
            )
        partial.replace(folder / "trace.jsonl")
        inference.pop("prediction_logs")
        inference["trace_sha256"] = sha256_file(folder / "trace.jsonl")
        write_json(summary_path, inference)
        write_json(folder / "progress.json", {"status": "COMPLETE", "sampled_frames": count})
        return inference
    except BaseException as exc:
        write_json(
            folder / "progress.json",
            {
                "status": "INTERRUPTED",
                "error": repr(exc),
                "sampled_frames": count,
                "partial_trace": str(partial),
                "resume_policy": "RESTART_INTERRUPTED_SESSION_PRESERVE_ATTEMPT",
            },
        )
        raise


def run(output: Path, stride: int = 15, limit: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    write_json(output / "cohort_manifest.json", cohort)
    if cohort["status"] == "BLOCKED_MISSING_DATA":
        write_json(output / "full_experiments.json", cohort)
        return cohort
    model_shas = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in (SPECIALIST_PATH, AUXILIARY_PATH, POSE_PATH)
    }
    matcher_path = ROOT / "training/dmd/temporal_eval.py"
    stage_paths = [
        ROOT / "training/dmd/run_autonomous_v20.py",
        ROOT / "training/dmd/run_verified_v22.py",
        ROOT / "training/dmd/action_adapter.py",
        ROOT / "backend/ai/auxiliary.py",
        ROOT / "backend/ai/events.py",
        ROOT / "training/dmd/driver_phone_association.py",
    ]
    protocol = {
        "scope": "SHORT_SMOKE_TEST" if limit is not None else SCOPE,
        "sample_stride": stride,
        "max_sampled_frames": limit,
        "models": model_shas,
        "config_sha256": sha256_file(CONFIG_PATH),
        "matcher_sha256": sha256_file(matcher_path),
        "policy_sha256": sha256_file(POLICY_PATH),
        "frozen_policy": read_json(POLICY_PATH),
        "cohort": cohort,
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in stage_paths},
        "stage1_signature": str(
            inspect.signature(AutonomousVideoInferenceEngine.run_session_inference)
        ),
        "B0": "Original V20 D5 implementation; legacy role/pose-first semantics retained",
        "B1": "D5 + conservative association, no verified seat calibration available; UNKNOWN",
        "B2": "B1 + explicitly RULE_ONLY; identical candidate-frame pose exposure",
        "B3": "B1 + verified V6; fail closed on provenance/causal timing",
        "B4": "B1 + HYBRID_RULE_V6; fail closed on provenance/causal timing",
        "changed_factor_warning": "B1 includes V21 100px/85px geometry; association-only effect cannot be isolated",
        "temporal_output_semantics": "ACTIVATION_ALERT_WINDOW; no finalized duration tracking or EOF tail extrapolation",
        "exposure_definition": "decoded source frames / container fps; sampled indices must exactly match stride grid",
        "resume_policy": "completed sessions only; interrupted trace preserved and session restarted",
        "data_rights": "local temporal development diagnostics only; no fitting, upload, approval or holdout access",
    }
    # Plan written before any new inference or scoring; never silently resume changed sources/config/code.
    protocol_path = output / "evaluation_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("RESUME_PROTOCOL_CHANGED; choose a new output directory")
    write_json(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)
    write_json(
        output / "full_experiment_manifest.json",
        {
            "protocol_sha256": protocol_sha,
            "runs": MODES,
            "status": "PREFLIGHT_COMPLETE",
            "started_unix_seconds": time.time(),
        },
    )
    config = read_json(CONFIG_PATH)
    frozen = read_json(POLICY_PATH)
    policy = TemporalMatchingPolicy(
        **{
            k: frozen[k]
            for k in ("iou_threshold", "onset_tolerance_seconds", "offset_tolerance_seconds")
        }
    )
    engine_kwargs = {
        k: v for k, v in config.items() if k in inspect.signature(TemporalEventEngine).parameters
    }
    results = {}
    matches = {}
    for run_id, mode in MODES.items():
        blockers = []
        if run_id in {"B3", "B4"}:
            for session in cohort["sessions"]:
                try:
                    ActionInferenceAdapter(
                        mode=mode,
                        heldout_subject=session["subject_id"],
                        missing_checkpoint_policy="ERROR",
                    )
                except CheckpointSafetyError as exc:
                    blockers.append({"subject_id": session["subject_id"], "reason": str(exc)})
        results[run_id] = {
            "requested_mode": mode,
            "status": "BLOCKED" if blockers else "PENDING",
            "blockers": blockers,
            "sessions": [],
        }
        if blockers:
            write_json(output / f"full_predictions_{run_id}.status.json", results[run_id])
            # Empty files carry no pseudo predictions; status sidecar is authoritative.
            (output / f"full_predictions_{run_id}.jsonl").touch(exist_ok=True)
    write_json(output / "full_experiments.json", results)
    # Shared raw inference is persisted once. Replay does not invoke detectors again.
    for session in cohort["sessions"]:
        sid = session["subject_id"]
        print(f"[V22] Extracting {sid}", flush=True)
        folder = output / "private" / sid
        extract_session(Path(session["video_path"]), folder, config, stride, limit)
    for run_id, mode in MODES.items():
        if results[run_id]["status"] == "BLOCKED":
            continue
        run_started = time.perf_counter()
        matches[run_id] = {}
        with (output / f"full_predictions_{run_id}.jsonl").open("w", encoding="utf-8") as handle:
            for session in cohort["sessions"]:
                sid = session["subject_id"]
                folder = output / "private" / sid
                inference = read_json(folder / "inference_summary.json")
                engine = TemporalEventEngine(**engine_kwargs)
                adapter = (
                    ActionInferenceAdapter(mode="RULE_ONLY", heldout_subject=sid)
                    if run_id == "B2"
                    else None
                )
                events, indices = [], []
                with (folder / "trace.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        indices.append(record["frame_idx"])
                        if run_id == "B0":
                            observation = Observation(**record["observation"])
                            detail = {
                                "legacy_role_assumption": True,
                                "candidates": record["candidates"],
                            }
                        else:
                            observation, detail = replay_verified_record(record, adapter)
                        emitted = engine.add(observation)
                        for event in emitted:
                            if event.event_type == "PHONE":
                                events.append(
                                    {
                                        "event_type": "PHONE",
                                        "start_seconds": event.start_timestamp,
                                        "end_seconds": event.end_timestamp,
                                        "confidence": event.confidence,
                                    }
                                )
                        handle.write(
                            json.dumps(
                                {
                                    "subject_id": sid,
                                    "session_id": session["session_id"],
                                    "camera_id": "BODY",
                                    "frame_idx": record["frame_idx"],
                                    "timestamp": observation.timestamp,
                                    "observation": asdict(observation),
                                    **detail,
                                    "emitted_events": [asdict(e) for e in emitted],
                                    "protocol_sha256": protocol_sha,
                                }
                            )
                            + "\n"
                        )
                events = deduplicate(events, config["cooldown_seconds"])
                if run_id == "B0":
                    # Refactor and replay parity check against events generated during actual inference.
                    original = inference["predicted_events"]
                    if [(e["start_seconds"], e["end_seconds"]) for e in events] != [
                        (e["start_seconds"], e["end_seconds"]) for e in original
                    ]:
                        raise ValueError("B0_REPLAY_PARITY_FAILED")
                scored = evaluate_predictions_stage2(
                    events,
                    Path(session["annotation_path"]),
                    session["fps"],
                    inference["total_frames"],
                    inference["total_seconds"],
                    policy,
                )
                gt = [
                    i.to_dict()
                    for i in parse_dmd_openlabel(
                        Path(session["annotation_path"]),
                        fps=session["fps"],
                        total_frames=inference["total_frames"],
                    ).phone_intervals
                ]
                matches[run_id][sid] = {
                    **scored,
                    "boundary_quality": boundary_metrics(scored["matched_events"], gt, events),
                }
                results[run_id]["sessions"].append(
                    {
                        "subject_id": sid,
                        "source_frames": session["source_frames"],
                        "fps": session["fps"],
                        "sample_stride": stride,
                        "frame_indices": indices,
                        "scope": protocol["scope"],
                        "evaluated_seconds": inference["total_seconds"],
                        "gt_count": scored["total_gt_events"],
                        "tp": scored["true_positives"],
                        "fp": scored["false_positives"],
                        "fn": scored["false_negatives"],
                        "inference_wall_seconds": inference["wall_time_seconds"],
                        "action_mode": adapter.provenance() if adapter else mode,
                    }
                )
        sessions = results[run_id]["sessions"]
        if limit is None:
            validate_full_coverage(sessions, cohort["subjects"], cohort["total_gt_events"])
        tp, fp, fn = (sum(s[k] for s in sessions) for k in ("tp", "fp", "fn"))
        minutes = sum(s["evaluated_seconds"] for s in sessions) / 60
        results[run_id].update(
            {
                "status": "COMPLETE" if limit is None else "SHORT_SMOKE_TEST",
                "scope": protocol["scope"],
                "subject_count": len(sessions),
                "session_count": len(sessions),
                "evaluated_frames": sum(len(s["frame_indices"]) for s in sessions),
                "evaluated_minutes": minutes,
                "gt_count": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "fa_per_minute": fp / minutes,
                "prediction_sha256": sha256_file(output / f"full_predictions_{run_id}.jsonl"),
                "protocol_sha256": protocol_sha,
                "model_sha256": model_shas,
                "config_sha256": protocol["config_sha256"],
                "matcher_sha256": protocol["matcher_sha256"],
                "replay_and_scoring_wall_seconds": time.perf_counter() - run_started,
            }
        )
        write_json(output / "full_experiments.json", results)
        write_json(output / "full_event_matches.json", matches)
    if (
        sha256_file(matcher_path) != protocol["matcher_sha256"]
        or sha256_file(POLICY_PATH) != protocol["policy_sha256"]
    ):
        raise ValueError("FROZEN_MATCHER_CHANGED")
    write_json(output / "event_matches.json", matches)
    write_json(output / "full_metrics_comparison.json", results)
    write_json(
        output / "raw_prediction_manifest.json",
        {
            r: {
                "path": f"full_predictions_{r}.jsonl",
                "sha256": sha256_file(output / f"full_predictions_{r}.jsonl"),
                "status": v["status"],
                "record_count": v.get("evaluated_frames", 0),
            }
            for r, v in results.items()
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--smoke-samples", type=int)
    args = parser.parse_args()
    if args.sample_stride <= 0 or (args.smoke_samples is not None and args.smoke_samples <= 0):
        parser.error("stride and smoke limit must be positive")
    result = run(args.output, args.sample_stride, args.smoke_samples)
    print(
        json.dumps(
            {
                k: {key: value for key, value in v.items() if key != "sessions"}
                for k, v in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
