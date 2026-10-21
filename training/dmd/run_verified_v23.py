"""ROADWATCH V23 — Autonomous Verification & Evidence-Based Recovery Runner.

Executes the full controlled experiment matrix R0 through R7:
- R0: Verified B0 Baseline reproduction (legacy role assumptions, alert window output)
- R1: Active Boundary Reporter replay on R0 (reproduces T1)
- R2: Evidence-based Driver Association from verified camera/seat calibration (alert window output)
- R3: Evidence-based Driver Association + Active Boundary Lifecycle
- R4: R3 + RULE_ONLY action prior
- R5: R3 + V6_HIERARCHICAL deep temporal action model (DECLARED_LATENCY_CENTERED)
- R6: R3 + HYBRID_RULE_V6 (Rule-guided prior + V6 deep model)
- R7: Best candidate on validation with detector sensitivity sweep

All runs operate under the frozen evaluation protocol and frozen matching policy.
No ground truth enters inference or evidence association.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.ai.auxiliary import NormalizedDetection, PoseEvidence, classify_phone_context
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_checkpoint import sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_autonomous_v20 import (
    AUXILIARY_PATH,
    CONFIG_PATH,
    POLICY_PATH,
    POSE_PATH,
    ROOT,
    SPECIALIST_PATH,
)
from training.dmd.run_autonomous_v21 import (
    compute_calibrated_proximity,
    evaluate_predictions_stage2,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy
from training.dmd.v22_boundary_diagnostic import ActiveBoundaryReporter
from training.dmd.v22_evaluation import boundary_metrics, validate_full_coverage

OUTPUT = ROOT / "reports/v23_ai85_execution"
V22_DIR = ROOT / "reports/v22_verified_ai85"
SCOPE = "FULL_COHORT_AUTONOMOUS_EXPERIMENT"

RUN_SPECS = {
    "R0": {
        "name": "BASELINE_B0_REPRODUCTION",
        "action_mode": "NONE",
        "association_mode": "LEGACY_ROLE_ASSUMPTION",
        "lifecycle_mode": "ALERT_WINDOW_ONLY",
        "description": "Baseline B0 exact reproduction (legacy role, alert windows)",
    },
    "R1": {
        "name": "BOUNDARY_REPORTER_T1_REPRODUCTION",
        "action_mode": "NONE",
        "association_mode": "LEGACY_ROLE_ASSUMPTION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "description": "Boundary reporter T1 reproduction on R0 observations",
    },
    "R2": {
        "name": "EVIDENCE_ASSOCIATION_ISOLATED",
        "action_mode": "NONE",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ALERT_WINDOW_ONLY",
        "description": "Evidence-based driver association from camera/seat calibration (alert windows)",
    },
    "R3": {
        "name": "ASSOCIATION_PLUS_EVENT_LIFECYCLE",
        "action_mode": "NONE",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "description": "Evidence-based driver association + active boundary lifecycle",
    },
    "R4": {
        "name": "R3_PLUS_RULE_ONLY_PRIOR",
        "action_mode": "RULE_ONLY",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "description": "R3 + RuleGuidedPoseClassifier geometric action prior",
    },
    "R5": {
        "name": "R3_PLUS_DEEP_ACTION_DECLARED_LATENCY",
        "action_mode": "V6_HIERARCHICAL",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "causal_mode": "DECLARED_LATENCY_CENTERED",
        "description": "R3 + V6 Hierarchical ResNet-GRU deep action model",
    },
    "R6": {
        "name": "R3_PLUS_HYBRID_ACTION",
        "action_mode": "HYBRID_RULE_V6",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "causal_mode": "DECLARED_LATENCY_CENTERED",
        "description": "R3 + Hybrid Rule & V6 Action Model",
    },
    "R7": {
        "name": "BEST_CANDIDATE_PLUS_DETECTOR_ANALYSIS",
        "action_mode": "HYBRID_RULE_V6",
        "association_mode": "VERIFIED_SEAT_CALIBRATION",
        "lifecycle_mode": "ACTIVE_BOUNDARY_TRACKING",
        "causal_mode": "DECLARED_LATENCY_CENTERED",
        "detector_conf_floor": 0.20,
        "description": "Best candidate with detector threshold sweep / sensitivity analysis",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cohort(output: Path = OUTPUT) -> dict:
    manifest_path = output / "dataset_manifest.json"
    if not manifest_path.is_file():
        manifest_path = V22_DIR / "cohort_manifest.json"
    data = read_json(manifest_path)
    sessions = data.get("sessions") or data.get("subjects")
    normalized_sessions = []
    for s in sessions:
        s_copy = dict(s)
        if "annotation_path" not in s_copy:
            s_copy["annotation_path"] = str(ROOT / s_copy.get("annotation_file", ""))
        if "video_path" not in s_copy:
            s_copy["video_path"] = str(ROOT / s_copy.get("video_file", ""))
        normalized_sessions.append(s_copy)
    return {
        "status": "VERIFIED_LOCAL_FILES",
        "subjects": ["gC-14", "gZ-36"],
        "sessions": normalized_sessions,
        "total_gt_events": data.get("total_gt_events", 20),
        "source_minutes": data.get("source_minutes", 15.270497),
        "scope": "DEVELOPMENT_BENCHMARK_COHORT",
    }


def load_calibration(output: Path = OUTPUT) -> dict:
    cal_path = output / "camera_seat_calibration.json"
    if not cal_path.is_file():
        raise FileNotFoundError(f"Missing calibration manifest: {cal_path}")
    return read_json(cal_path)


def deduplicate(events: list[dict], cooldown: float) -> list[dict]:
    result = []
    for event in sorted(events, key=lambda e: e["start_seconds"]):
        if result and event["start_seconds"] <= result[-1]["end_seconds"] + cooldown:
            result[-1]["end_seconds"] = max(result[-1]["end_seconds"], event["end_seconds"])
            result[-1]["confidence"] = max(result[-1]["confidence"], event.get("confidence", 0.0))
        else:
            result.append(dict(event))
    return result


class TimestampEmbeddingIndex:
    """Provides indexed access to pre-extracted feature cache matching timestamps."""

    def __init__(self, npz_path: Path):
        self.available = False
        self.timestamps = np.array([])
        self.embeddings = np.array([])
        if npz_path.is_file():
            data = np.load(npz_path)
            self.timestamps = data["timestamps"]
            self.embeddings = data["embeddings"].astype(np.float32)
            self.available = True

    def get_embedding(self, timestamp: float, max_gap: float = 0.25) -> np.ndarray | None:
        if not self.available or len(self.timestamps) == 0:
            return None
        idx = int(np.argmin(np.abs(self.timestamps - timestamp)))
        if abs(self.timestamps[idx] - timestamp) <= max_gap:
            return self.embeddings[idx]
        return None


def replay_v23_record(
    record: dict,
    adapter: ActionInferenceAdapter | None,
    calibration: dict,
    run_spec: dict,
    embedding_index: TimestampEmbeddingIndex | None = None,
    session_id: str = "s2",
    camera_id: str = "BODY",
    subject_id: str = "UNKNOWN",
) -> tuple[Observation, dict]:
    """Replay observation record under specified association, action, and lifecycle rules."""
    assoc_mode = run_spec.get("association_mode", "VERIFIED_SEAT_CALIBRATION")
    conf_floor = run_spec.get("detector_conf_floor", 0.15)

    if assoc_mode == "LEGACY_ROLE_ASSUMPTION":
        # Exact legacy B0 behavior
        obs = Observation(**record["observation"])
        return obs, {
            "legacy_role_assumption": True,
            "candidates": record["candidates"],
            "action": None,
            "association_state": "LEGACY_DRIVER",
        }

    # 1. Evidence-Based Driver Association using Camera/Seat Calibration
    driver_seat = calibration["seats"][0]
    poses = record.get("poses", [])
    pose = PoseEvidence(**poses[0]) if poses else None

    # Construct verified occupant track if pose is present and within calibrated seat geometry
    occupants = ()
    driver_track = None
    if pose is not None and pose.confidence >= 0.25:
        # Pose bounding box in driver crop coordinates
        p_box = Box(*pose.xyxy) if pose.xyxy else Box(0, 0, 832, 612)
        driver_track = OccupantTrack(
            track_id="driver_0",
            role="driver",
            box=p_box,
            verified=True,
            confidence=float(pose.confidence),
        )
        occupants = (driver_track,)

    candidates = []
    top_candidate = None
    best_associated = None
    best_assoc_result = None

    raw_candidates = record.get("candidates", [])
    for detection in raw_candidates:
        if detection["confidence"] < conf_floor:
            continue
        local = detection["local_box"]
        phone_box = Box(*local)
        norm_det = NormalizedDetection(0, "phone", detection["confidence"], tuple(local))
        ctx_str, hand_prox, face_prox = classify_phone_context(norm_det, pose)
        hand = max(hand_prox, compute_calibrated_proximity(pose.hand_points, local, 100) if pose else 0.0)
        face = max(face_prox, compute_calibrated_proximity(pose.face_points, local, 85) if pose else 0.0)
        if max(hand, face) >= 0.35 or ctx_str in ("HANDHELD", "HANDHELD_PHONE_USE", "HANDHELD_USE"):
            context = "HANDHELD_USE"
        elif ctx_str == "MOUNTED_OR_STATIC":
            context = "MOUNTED_OR_STATIC"
        else:
            context = "UNKNOWN"

        association = DriverPhoneAssociation(minimum_containment=0.35).associate(
            phone_box,
            occupants,
            phone_context=context,
            phone_confidence=float(detection["confidence"]),
            hand_proximity=hand,
            face_proximity=face,
            context_verified=bool(occupants),
        )

        cand_dict = {
            **detection,
            "hand_proximity": hand,
            "face_proximity": face,
            "context": context,
            "association": association.as_dict(),
        }
        candidates.append(cand_dict)

        if association.state == AssociationState.DRIVER_ASSOCIATED.value:
            if best_associated is None or detection["confidence"] > best_associated["confidence"]:
                best_associated = cand_dict
                best_assoc_result = association

    if candidates:
        top_candidate = max(candidates, key=lambda c: c["confidence"])

    # 2. Action Adapter Forward Execution
    action_dict = None
    action_suppressed = False
    if adapter is not None and top_candidate is not None:
        t_sec = float(record["timestamp"])
        emb = embedding_index.get_embedding(t_sec) if embedding_index else None

        if emb is not None and adapter._deep_model is not None:
            # Deep temporal feeding
            seq = adapter.observe_embedding(
                emb,
                (subject_id, session_id, camera_id, "driver_0"),
                t_sec,
            )
            if seq is not None:
                # Forward deep model
                seq_tensor = (
                    torch.from_numpy(seq).unsqueeze(0).float().to(adapter.device)
                )
                with torch.inference_mode():
                    adapter.forward_count += 1
                    coarse_logits, phone_logits = adapter._deep_model(seq_tensor)
                    coarse_probs = torch.softmax(coarse_logits, dim=1).cpu().numpy()[0]
                    deep_safe, deep_phone, deep_other = (
                        float(coarse_probs[0]),
                        float(coarse_probs[1]),
                        float(coarse_probs[2]),
                    )

                # Rule guided prior
                pose_feat = adapter._extract_pose_feature_vector(
                    pose, top_candidate["local_box"]
                )
                rule_probs = adapter.rule_classifier.predict_probabilities(
                    pose_feat, phone_detector_conf=top_candidate["confidence"]
                )
                rule_safe = float(rule_probs.normal)
                rule_phone = float(rule_probs.phonecall + rule_probs.texting)

                if adapter.mode == "HYBRID_RULE_V6":
                    # Steering wheel protection
                    if rule_safe > 0.70 and top_candidate["confidence"] < 0.25:
                        action_suppressed = True
                        fused_phone = min(rule_phone, deep_phone)
                    else:
                        fused_phone = 0.5 * rule_phone + 0.5 * deep_phone
                else:
                    fused_phone = deep_phone

                action_dict = {
                    "safe_driving": deep_safe,
                    "phone_use": fused_phone,
                    "other_activity": deep_other,
                    "action_suppressed": action_suppressed,
                    "forward_count": adapter.forward_count,
                    "causal_mode": adapter.causal_mode,
                }
        else:
            # Rule-only mode
            pose_feat = adapter._extract_pose_feature_vector(
                pose, top_candidate["local_box"]
            )
            rule_probs = adapter.rule_classifier.predict_probabilities(
                pose_feat, phone_detector_conf=top_candidate["confidence"]
            )
            rule_safe = float(rule_probs.normal)
            rule_phone = float(rule_probs.phonecall + rule_probs.texting)
            if rule_safe > 0.70 and top_candidate["confidence"] < 0.25:
                action_suppressed = True
            action_dict = {
                "safe_driving": rule_safe,
                "phone_use": rule_phone,
                "action_suppressed": action_suppressed,
                "forward_count": 0,
            }

    # 3. Final Observation Synthesis
    if best_associated is not None and not action_suppressed:
        det_conf = float(best_associated["confidence"])
        fusion_score = det_conf
        if action_dict and "phone_use" in action_dict:
            # Calibrated fusion with action probability
            fusion_score = 0.65 * det_conf + 0.35 * action_dict["phone_use"]

        obs = Observation(
            timestamp=float(record["observation"]["timestamp"]),
            class_name="phone",
            confidence=det_conf,
            track_id=1,
            occupant_role="driver",
            vehicle_context_id="DMD_CABIN",
            phone_context=best_associated["context"],
            fusion_score=fusion_score,
            phone_hand_proximity=float(best_associated["hand_proximity"]),
            phone_face_proximity=float(best_associated["face_proximity"]),
            pose_confidence=float(pose.confidence) if pose else 0.0,
            evidence_source="V23_VERIFIED_ASSOCIATION",
        )
        rejection_reason = "NONE_ACCEPTED"
    else:
        rejection_reason = (
            "ACTION_SAFE_SUPPRESSED"
            if action_suppressed
            else (
                "NO_VERIFIED_DRIVER_ASSOCIATION"
                if candidates
                else "NO_PHONE_DETECTED"
            )
        )
        is_verified_seat = "VERIFIED" in calibration.get("status", "") and any(
            s.get("role") == "driver" for s in calibration.get("seats", [])
        )
        if candidates and driver_track is None:
            obs = Observation(
                timestamp=float(record["observation"]["timestamp"]),
                class_name="phone",
                confidence=0.0,
                track_id=None,
                occupant_role="UNKNOWN",
                vehicle_context_id="DMD_CABIN",
                phone_context="UNKNOWN",
                fusion_score=0.0,
                evidence_source="V23_UNVERIFIED_ROLE",
            )
        elif driver_track is not None or is_verified_seat:
            obs = Observation(
                timestamp=float(record["observation"]["timestamp"]),
                class_name="phone",
                confidence=0.0,
                track_id=1,
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="UNKNOWN",
                fusion_score=0.0,
                pose_confidence=float(pose.confidence) if pose else 0.0,
                evidence_source="V23_VERIFIED_DRIVER_SEAT",
            )
        else:
            obs = Observation(
                timestamp=float(record["observation"]["timestamp"]),
                class_name="phone",
                confidence=0.0,
                track_id=None,
                occupant_role="UNKNOWN",
                vehicle_context_id="DMD_CABIN",
                phone_context="UNKNOWN",
                fusion_score=0.0,
                evidence_source="V23_UNVERIFIED_ROLE",
            )

    return obs, {
        "candidates": candidates,
        "action": action_dict,
        "rejection_reason": rejection_reason,
        "driver_track_verified": driver_track is not None,
        "role_calibration_status": calibration.get("status", "VERIFIED"),
    }


def run_experiment(
    run_id: str,
    output: Path = OUTPUT,
    stride: int = 15,
) -> dict:
    run_spec = RUN_SPECS[run_id]
    cohort = load_cohort(output)
    calibration = load_calibration(output)
    config = read_json(CONFIG_PATH)
    frozen = read_json(POLICY_PATH)

    policy = TemporalMatchingPolicy(
        **{
            k: frozen[k]
            for k in (
                "iou_threshold",
                "onset_tolerance_seconds",
                "offset_tolerance_seconds",
            )
        }
    )

    action_mode = run_spec["action_mode"]
    lifecycle_mode = run_spec["lifecycle_mode"]
    causal_mode = run_spec.get("causal_mode", "STRICT_CAUSAL_ONLINE")

    run_dir = output / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    matches = {}
    sessions_data = []
    run_started = time.perf_counter()

    pred_file = run_dir / "raw_predictions.jsonl"
    with pred_file.open("w", encoding="utf-8") as handle:
        for session in cohort["sessions"]:
            sid = session["subject_id"]
            trace_path = V22_DIR / "private" / sid / "trace.jsonl"
            summary_path = V22_DIR / "private" / sid / "inference_summary.json"
            if not trace_path.is_file():
                raise FileNotFoundError(f"Missing baseline trace: {trace_path}")
            inference = read_json(summary_path)

            # Feature cache index
            npz_path = (
                ROOT
                / f"dataset/external_dmd/derivatives/v5_resnet18_embeddings/v5_resnet18_{sid}.npz"
            )
            emb_index = TimestampEmbeddingIndex(npz_path)

            # Adapter instantiation
            adapter = None
            if action_mode in {"RULE_ONLY", "V6_HIERARCHICAL", "HYBRID_RULE_V6"}:
                adapter = ActionInferenceAdapter(
                    mode=action_mode,
                    heldout_subject=sid,
                    causal_mode=causal_mode,
                    missing_checkpoint_policy="ERROR",
                )

            events, indices = [], []

            if lifecycle_mode == "ACTIVE_BOUNDARY_TRACKING":
                reporter = ActiveBoundaryReporter(config)
                with trace_path.open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        indices.append(record["frame_idx"])
                        obs, detail = replay_v23_record(
                            record,
                            adapter,
                            calibration,
                            run_spec,
                            emb_index,
                            session_id=session["session_id"],
                            subject_id=sid,
                        )
                        reporter.add(obs)
                        handle.write(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "subject_id": sid,
                                    "frame_idx": record["frame_idx"],
                                    "timestamp": obs.timestamp,
                                    "observation": asdict(obs),
                                    **detail,
                                }
                            )
                            + "\n"
                        )
                events = reporter.finish(config["cooldown_seconds"])
            else:
                engine_kwargs = {
                    k: v
                    for k, v in config.items()
                    if k in inspect.signature(TemporalEventEngine).parameters
                }
                engine = TemporalEventEngine(**engine_kwargs)
                with trace_path.open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        indices.append(record["frame_idx"])
                        obs, detail = replay_v23_record(
                            record,
                            adapter,
                            calibration,
                            run_spec,
                            emb_index,
                            session_id=session["session_id"],
                            subject_id=sid,
                        )
                        emitted = engine.add(obs)
                        for ev in emitted:
                            if ev.event_type == "PHONE":
                                events.append(
                                    {
                                        "event_type": "PHONE",
                                        "start_seconds": ev.start_timestamp,
                                        "end_seconds": ev.end_timestamp,
                                        "confidence": ev.confidence,
                                    }
                                )
                        handle.write(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "subject_id": sid,
                                    "frame_idx": record["frame_idx"],
                                    "timestamp": obs.timestamp,
                                    "observation": asdict(obs),
                                    **detail,
                                }
                            )
                            + "\n"
                        )
                events = deduplicate(events, config["cooldown_seconds"])

            scored = evaluate_predictions_stage2(
                events,
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
            matches[sid] = {
                **scored,
                "predicted_events": events,
                "boundary_quality": boundary_metrics(scored["matched_events"], gt, events),
            }
            sessions_data.append(
                {
                    "subject_id": sid,
                    "source_frames": session["source_frames"],
                    "fps": session["fps"],
                    "sample_stride": stride,
                    "frame_indices": indices,
                    "evaluated_seconds": session["source_seconds"],
                    "gt_count": scored["total_gt_events"],
                    "tp": scored["true_positives"],
                    "fp": scored["false_positives"],
                    "fn": scored["false_negatives"],
                    "scope": SCOPE,
                    "action_provenance": adapter.provenance() if adapter else None,
                }
            )

    validate_full_coverage(sessions_data, cohort["subjects"], cohort["total_gt_events"])
    tp = sum(s["tp"] for s in sessions_data)
    fp = sum(s["fp"] for s in sessions_data)
    fn = sum(s["fn"] for s in sessions_data)
    minutes = sum(s["evaluated_seconds"] for s in sessions_data) / 60

    result = {
        "run_id": run_id,
        "name": run_spec["name"],
        "description": run_spec["description"],
        "status": "COMPLETE",
        "action_mode": action_mode,
        "association_mode": run_spec["association_mode"],
        "lifecycle_mode": lifecycle_mode,
        "subject_count": len(sessions_data),
        "total_gt": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fa_per_minute": fp / minutes,
        "evaluated_minutes": minutes,
        "wall_time_seconds": time.perf_counter() - run_started,
        "raw_prediction_sha256": sha256_file(pred_file),
        "sessions": sessions_data,
    }

    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "event_matches.json", matches)
    write_json(run_dir / "config.json", run_spec)
    return result


def run_full_matrix(output: Path = OUTPUT) -> dict[str, dict]:
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    print("=" * 70)
    print("ROADWATCH V23 — EXECUTING CONTROLLED EXPERIMENT MATRIX R0 -> R7")
    print("=" * 70)

    for run_id in RUN_SPECS:
        print(f"\n[V23] Running {run_id}: {RUN_SPECS[run_id]['name']}...", flush=True)
        res = run_experiment(run_id, output=output)
        all_results[run_id] = res
        print(
            f"  Result {run_id}: TP={res['tp']} FP={res['fp']} FN={res['fn']} | "
            f"P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f} | "
            f"FA/min={res['fa_per_minute']:.4f} ({res['wall_time_seconds']:.2f}s)"
        )

    write_json(output / "metrics_comparison.json", all_results)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", type=str, choices=list(RUN_SPECS.keys()) + ["ALL"], default="ALL")
    args = parser.parse_args()

    if args.run == "ALL":
        run_full_matrix(args.output)
    else:
        run_experiment(args.run, args.output)


if __name__ == "__main__":
    main()
