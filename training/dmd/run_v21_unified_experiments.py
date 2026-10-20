"""ROADWATCH V21 — Unified Controlled Scientific Experiments & Full Ledger Generation.

Executes the primary experiments across the authorized 15.27 minute development cohort (gC-14, gZ-36):
- EXP-0: Exact V20 Reproduction Baseline (Hardcoded driver role, legacy whole-body proximity, no action model)
- EXP-1: Verified Occupant Association + Calibrated Anatomical Proximity (Strict driver role, 100px edge proximity, no action model)
- EXP-2: Trained Action Model Integration (Driver association + calibrated proximity + Hybrid Rule-Guided V6 ResNet-GRU action gating)

Extracts video features once and streams to parallel temporal event engines, guaranteeing
identical frame exposure and timing across all experiments.

Outputs all required machine-readable JSON/JSONL deliverables to reports/v21_ai85_recovery/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.ai.auxiliary import PoseEstimator, PoseEvidence, _point_box_proximity
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_adapter import ActionInferenceAdapter, ActionScores
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationResult,
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_autonomous_v21 import (
    box_iou,
    compute_calibrated_proximity,
    evaluate_predictions_stage2,
    merge_detections,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "v21_ai85_recovery"

SPECIALIST_PATH = MODELS_DIR / "active" / "v2" / "phone_detector.pt"
AUXILIARY_PATH = MODELS_DIR / "auxiliary" / "yolo11n.pt"
POSE_PATH = MODELS_DIR / "auxiliary" / "yolo11n-pose.pt"
CONFIG_PATH = MODELS_DIR / "temporal_config_v2.json"
POLICY_PATH = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"


def run_unified_v21_experiments(
    cohort: str = "two-subject",
    sample_stride: int = 15,
    max_frames: int | None = None,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    temporal_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy_data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    matching_policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    targets_dict = {
        "gC-14": (
            ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4",
            ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json",
        ),
        "gZ-36": (
            ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4",
            ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json",
        ),
        "gB-9": (
            ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_body.mp4",
            ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_ann_distraction.json",
        ),
    }

    active_cohort = ["gB-9", "gC-14", "gZ-36"] if cohort == "three-subject" else ["gC-14", "gZ-36"]
    targets = [(sub, targets_dict[sub][0], targets_dict[sub][1]) for sub in active_cohort if targets_dict[sub][0].exists()]

    # Load shared detection and pose models
    print("[INIT] Loading Specialist, Auxiliary, Pose and Action models...", flush=True)
    specialist = YOLO(str(SPECIALIST_PATH))
    auxiliary = YOLO(str(AUXILIARY_PATH))
    pose_estimator = PoseEstimator(weights=str(POSE_PATH), confidence=0.25)
    association_engine = DriverPhoneAssociation(minimum_containment=0.35, ambiguity_margin=0.15)

    engine_supported = {
        "window_seconds", "min_positive_seconds", "min_observations", "cooldown_seconds",
        "smoothing_alpha", "feature_positive_score", "positive_ratio", "gap_tolerance_seconds",
        "candidate_threshold", "activation_threshold", "release_threshold", "occlusion_bridge_seconds",
    }
    engine_kwargs = {k: v for k, v in temporal_cfg.items() if k in engine_supported}

    # Tracking records across experiments
    exp_results: dict[str, dict[str, Any]] = {
        "EXP-0_V20_BASELINE": {"subjects": {}, "predicted_events": []},
        "EXP-1_DRIVER_ASSOCIATION": {"subjects": {}, "predicted_events": []},
        "EXP-2_ACTION_MODEL": {"subjects": {}, "predicted_events": []},
    }

    all_raw_predictions = []
    all_candidate_records = []
    all_action_logs = []
    stage_rejections = []

    for sub_id, vid_path, ann_path in targets:
        print(f"\n=======================================================", flush=True)
        print(f"PROCESSING SESSION: {sub_id} ({vid_path.name})", flush=True)
        print(f"=======================================================", flush=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {vid_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
        full_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        action_adapter = ActionInferenceAdapter(mode="HYBRID_RULE_V6", heldout_subject=sub_id)

        # 3 Parallel engines for this session
        engine_exp0 = TemporalEventEngine(**engine_kwargs)
        engine_exp1 = TemporalEventEngine(**engine_kwargs)
        engine_exp2 = TemporalEventEngine(**engine_kwargs)

        preds_exp0: list[dict[str, Any]] = []
        preds_exp1: list[dict[str, Any]] = []
        preds_exp2: list[dict[str, Any]] = []

        frame_idx = 0
        processed_count = 0
        t_start = time.perf_counter()

        static_roi_norm = (0.0, 0.10, 0.65, 0.95)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]
            current_frame_idx = frame_idx
            timestamp = current_frame_idx / fps
            frame_idx += 1

            if current_frame_idx % sample_stride != 0:
                continue

            processed_count += 1
            if max_frames and processed_count > max_frames:
                break

            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                rate = processed_count / elapsed if elapsed > 0 else 0.0
                print(f"  [{sub_id}] Sampled {processed_count} frames ({round(timestamp, 1)}s video / {round(elapsed, 1)}s wall) [{round(rate, 2)} fps]...", flush=True)

            # 1. Driver Crop
            rx1 = max(0, int(static_roi_norm[0] * w))
            ry1 = max(0, int(static_roi_norm[1] * h))
            rx2 = min(w, int(static_roi_norm[2] * w))
            ry2 = min(h, int(static_roi_norm[3] * h))
            driver_crop = frame[ry1:ry2, rx1:rx2]

            # 2. Pose
            poses = pose_estimator.predict(driver_crop)
            driver_pose: PoseEvidence | None = None
            occupant_tracks: list[OccupantTrack] = []
            if poses:
                best_pose = max(poses, key=lambda p: p.confidence)
                if best_pose.confidence >= 0.25:
                    driver_pose = best_pose
                    pbox = best_pose.xyxy or (0.0, 0.0, float(rx2 - rx1), float(ry2 - ry1))
                    occupant_tracks.append(
                        OccupantTrack(
                            track_id="driver_0",
                            role="driver",
                            box=Box(pbox[0], pbox[1], pbox[2], pbox[3]),
                            verified=True,
                            confidence=float(best_pose.confidence),
                        )
                    )

            # 3. Detection (Specialist + Auxiliary in ROI)
            spec_res = specialist.predict(driver_crop, imgsz=640, conf=0.15, device="cpu", verbose=False)[0]
            raw_dets = []
            for b in spec_res.boxes:
                conf = float(b.conf.item())
                loc_xyxy = [float(v) for v in b.xyxy[0].cpu().tolist()]
                glob_xyxy = [loc_xyxy[0] + rx1, loc_xyxy[1] + ry1, loc_xyxy[2] + rx1, loc_xyxy[3] + ry1]
                raw_dets.append({"box": glob_xyxy, "confidence": conf, "source": "specialist_roi", "local_box": loc_xyxy})

            aux_res = auxiliary.predict(driver_crop, imgsz=640, conf=0.15, classes=[67], device="cpu", verbose=False)[0]
            for b in aux_res.boxes:
                conf = float(b.conf.item())
                loc_xyxy = [float(v) for v in b.xyxy[0].cpu().tolist()]
                glob_xyxy = [loc_xyxy[0] + rx1, loc_xyxy[1] + ry1, loc_xyxy[2] + rx1, loc_xyxy[3] + ry1]
                raw_dets.append({"box": glob_xyxy, "confidence": conf, "source": "auxiliary_roi", "local_box": loc_xyxy})

            merged_dets = merge_detections(raw_dets, iou_threshold=0.50) if len(raw_dets) > 1 else raw_dets

            # 4. Action Recognition
            primary_box = merged_dets[0]["local_box"] if merged_dets else None
            primary_conf = merged_dets[0]["confidence"] if merged_dets else 0.0
            action_scores = action_adapter.predict(
                driver_crop=driver_crop,
                pose=driver_pose,
                phone_conf=primary_conf,
                phone_box_local=primary_box,
            )
            all_action_logs.append({
                "subject_id": sub_id,
                "timestamp": round(timestamp, 3),
                "frame_idx": current_frame_idx,
                **action_scores.to_dict(),
            })

            # 5. Evaluate Observations for each experiment policy
            if not merged_dets:
                obs_empty = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=0.0,
                    track_id=(occupant_tracks[0].track_id if occupant_tracks else None),
                    occupant_role="UNKNOWN",
                    vehicle_context_id="DMD_CABIN",
                    phone_context="UNKNOWN",
                    fusion_score=0.0,
                    evidence_source="AUTONOMOUS_V21",
                )
                evs0 = engine_exp0.add(obs_empty)
                evs1 = engine_exp1.add(obs_empty)
                evs2 = engine_exp2.add(obs_empty)
            else:
                # Top detection by raw confidence (V20 baseline selection)
                top_v20 = max(merged_dets, key=lambda d: d["confidence"])
                loc_v20 = top_v20["local_box"]

                # Legacy proximity (whole-body diagonal)
                leg_h_prox = 0.0
                leg_f_prox = 0.0
                if driver_pose:
                    pbox = driver_pose.xyxy or (0.0, 0.0, float(rx2 - rx1), float(ry2 - ry1))
                    leg_h_prox = _point_box_proximity(driver_pose.hand_points, tuple(loc_v20), pbox)
                    leg_f_prox = _point_box_proximity(driver_pose.face_points, tuple(loc_v20), pbox)

                leg_context = "HANDHELD" if (leg_h_prox >= 0.55 or leg_f_prox >= 0.55) else "UNKNOWN"
                obs_exp0 = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=top_v20["confidence"],
                    track_id="driver_0",
                    occupant_role="driver",  # Hardcoded in V20
                    vehicle_context_id="DMD_CABIN",
                    phone_context=leg_context,
                    fusion_score=top_v20["confidence"],
                    phone_hand_proximity=leg_h_prox,
                    phone_face_proximity=leg_f_prox,
                    pose_confidence=(driver_pose.confidence if driver_pose else 0.0),
                    evidence_source="V20_BASELINE",
                )
                evs0 = engine_exp0.add(obs_exp0)

                # V21 Multi-candidate evaluation & verified association
                v21_candidates = []
                for det in merged_dets:
                    lbox = det["local_box"]
                    p_box = Box(lbox[0], lbox[1], lbox[2], lbox[3])
                    h_prox = compute_calibrated_proximity(driver_pose.hand_points, lbox, radius_pixels=100.0) if driver_pose else 0.0
                    f_prox = compute_calibrated_proximity(driver_pose.face_points, lbox, radius_pixels=85.0) if driver_pose else 0.0

                    p_context = "UNKNOWN"
                    if h_prox >= 0.35 or f_prox >= 0.35:
                        p_context = "HANDHELD_USE"
                    elif det["confidence"] >= 0.30:
                        p_context = "MOUNTED_OR_STATIC"

                    assoc_res = association_engine.associate(
                        p_box,
                        tuple(occupant_tracks),
                        phone_context=p_context,
                        phone_confidence=det["confidence"],
                        hand_proximity=h_prox,
                        face_proximity=f_prox,
                        context_verified=(p_context == "HANDHELD_USE"),
                    )

                    cand_dict = {
                        "det": det,
                        "hand_prox": h_prox,
                        "face_prox": f_prox,
                        "context": p_context,
                        "association": assoc_res,
                    }
                    v21_candidates.append(cand_dict)

                    all_candidate_records.append({
                        "subject_id": sub_id,
                        "timestamp": round(timestamp, 3),
                        "frame_idx": current_frame_idx,
                        "box": [round(v, 1) for v in det["box"]],
                        "confidence": round(det["confidence"], 4),
                        "source": det["source"],
                        "hand_prox": round(h_prox, 3),
                        "face_prox": round(f_prox, 3),
                        "assoc_state": assoc_res.state,
                        "reasons": list(assoc_res.reason_codes),
                    })

                # Candidate Ranking
                def cand_rank(c: dict[str, Any]) -> float:
                    is_drv = 2.0 if c["association"].state == AssociationState.DRIVER_ASSOCIATED.value else 0.0
                    is_hh = 1.5 if c["context"] == "HANDHELD_USE" else 0.0
                    geom = max(c["hand_prox"], c["face_prox"])
                    return is_drv + is_hh + geom + c["det"]["confidence"]

                best_cand = max(v21_candidates, key=cand_rank)
                top_v21_det = best_cand["det"]
                h_prox = best_cand["hand_prox"]
                f_prox = best_cand["face_prox"]
                assoc_state = best_cand["association"].state

                # --- Policy EXP-1: Verified Association ON, Action Gating OFF ---
                admit_exp1 = True
                rej_exp1 = None
                if best_cand["context"] != "HANDHELD_USE":
                    admit_exp1 = False
                    rej_exp1 = "PHONE_VISIBLE_WITHOUT_HANDHELD_USE"
                elif assoc_state != AssociationState.DRIVER_ASSOCIATED.value:
                    admit_exp1 = False
                    rej_exp1 = f"EXCLUSION_{assoc_state}"

                obs_exp1 = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=(top_v21_det["confidence"] if admit_exp1 else 0.0),
                    track_id=(occupant_tracks[0].track_id if occupant_tracks else None),
                    occupant_role=("driver" if admit_exp1 else "UNKNOWN"),
                    vehicle_context_id="DMD_CABIN",
                    phone_context=("HANDHELD_USE" if admit_exp1 else "UNKNOWN"),
                    fusion_score=(top_v21_det["confidence"] if admit_exp1 else 0.0),
                    phone_hand_proximity=h_prox,
                    phone_face_proximity=f_prox,
                    pose_confidence=(driver_pose.confidence if driver_pose else 0.0),
                    evidence_source="EXP-1_ASSOCIATION",
                )
                evs1 = engine_exp1.add(obs_exp1)

                # --- Policy EXP-2: Verified Association ON, Action Gating ON ---
                admit_exp2 = admit_exp1
                fused_conf_exp2 = top_v21_det["confidence"]
                rej_exp2 = rej_exp1

                if admit_exp2 and action_scores is not None:
                    if action_scores.safe_driving > 0.65 and action_scores.phone_use < 0.20:
                        admit_exp2 = False
                        rej_exp2 = "ACTION_SAFE_DRIVING_SUPPRESSION"
                    elif action_scores.phone_use > 0.40 and fused_conf_exp2 >= 0.15:
                        fused_conf_exp2 = min(1.0, fused_conf_exp2 * 1.15)

                obs_exp2 = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=(fused_conf_exp2 if admit_exp2 else 0.0),
                    track_id=(occupant_tracks[0].track_id if occupant_tracks else None),
                    occupant_role=("driver" if admit_exp2 else "UNKNOWN"),
                    vehicle_context_id="DMD_CABIN",
                    phone_context=("HANDHELD_USE" if admit_exp2 else "UNKNOWN"),
                    fusion_score=(fused_conf_exp2 if admit_exp2 else 0.0),
                    phone_hand_proximity=h_prox,
                    phone_face_proximity=f_prox,
                    pose_confidence=(driver_pose.confidence if driver_pose else 0.0),
                    evidence_source="EXP-2_ACTION_FUSED",
                )
                evs2 = engine_exp2.add(obs_exp2)

                if rej_exp2:
                    stage_rejections.append({
                        "subject_id": sub_id,
                        "frame_idx": current_frame_idx,
                        "timestamp": round(timestamp, 3),
                        "reason": rej_exp2,
                        "detector_conf": round(top_v21_det["confidence"], 4),
                        "safe_driving_score": round(action_scores.safe_driving, 4),
                        "phone_use_score": round(action_scores.phone_use, 4),
                    })

                all_raw_predictions.append({
                    "subject_id": sub_id,
                    "frame_idx": current_frame_idx,
                    "timestamp": round(timestamp, 3),
                    "box": [round(v, 1) for v in top_v21_det["box"]],
                    "confidence": round(top_v21_det["confidence"], 4),
                    "detector_source": top_v21_det["source"],
                    "hand_prox": round(h_prox, 3),
                    "face_prox": round(f_prox, 3),
                    "assoc_state": assoc_state,
                    "admitted_exp1": admit_exp1,
                    "admitted_exp2": admit_exp2,
                    "rejection_exp2": rej_exp2,
                    "safe_score": round(action_scores.safe_driving, 3),
                    "phone_score": round(action_scores.phone_use, 3),
                })

            for e in evs0:
                if e.event_type == "PHONE":
                    preds_exp0.append({"event_type": "PHONE", "start_seconds": e.start_timestamp, "end_seconds": e.end_timestamp, "confidence": e.confidence, "duration": round(e.end_timestamp - e.start_timestamp, 2), "subject_id": sub_id})
            for e in evs1:
                if e.event_type == "PHONE":
                    preds_exp1.append({"event_type": "PHONE", "start_seconds": e.start_timestamp, "end_seconds": e.end_timestamp, "confidence": e.confidence, "duration": round(e.end_timestamp - e.start_timestamp, 2), "subject_id": sub_id})
            for e in evs2:
                if e.event_type == "PHONE":
                    preds_exp2.append({"event_type": "PHONE", "start_seconds": e.start_timestamp, "end_seconds": e.end_timestamp, "confidence": e.confidence, "duration": round(e.end_timestamp - e.start_timestamp, 2), "subject_id": sub_id})

        cap.release()
        actual_total_frames = frame_idx if max_frames else full_video_frames
        total_seconds = actual_total_frames / fps
        wall_time = time.perf_counter() - t_start
        eff_fps = round(processed_count / wall_time if wall_time else 0.0, 2)
        print(f"  [{sub_id}] Session Complete: {processed_count} sampled frames in {round(wall_time, 1)}s ({eff_fps} fps).", flush=True)

        # STAGE 2: Post-Hoc Evaluation on this session for each experiment
        for exp_key, raw_p_list in [("EXP-0_V20_BASELINE", preds_exp0), ("EXP-1_DRIVER_ASSOCIATION", preds_exp1), ("EXP-2_ACTION_MODEL", preds_exp2)]:
            # Cooldown deduplication
            cooldown = float(temporal_cfg.get("cooldown_seconds", 3.0))
            deduped = []
            for cand in sorted(raw_p_list, key=lambda x: x["start_seconds"]):
                if not deduped:
                    deduped.append(dict(cand))
                else:
                    last = deduped[-1]
                    if cand["start_seconds"] <= last["end_seconds"] + cooldown:
                        if cand["end_seconds"] > last["end_seconds"]:
                            last["end_seconds"] = cand["end_seconds"]
                            last["duration"] = round(last["end_seconds"] - last["start_seconds"], 2)
                    else:
                        deduped.append(dict(cand))

            exp_results[exp_key]["predicted_events"].extend(deduped)

            eval_res = evaluate_predictions_stage2(
                predicted_events=deduped,
                annotation_path=ann_path,
                fps=fps,
                total_frames=actual_total_frames,
                total_seconds=total_seconds,
                matching_policy=matching_policy,
            )
            eval_res["effective_fps"] = eff_fps
            eval_res["wall_time_seconds"] = round(wall_time, 2)
            exp_results[exp_key]["subjects"][sub_id] = eval_res
            print(f"    [{exp_key}] {sub_id} -> TP={eval_res['true_positives']}, FP={eval_res['false_positives']}, FN={eval_res['false_negatives']}, Prec={eval_res['precision']}, Rec={eval_res['recall']}, F1={eval_res['f1']}, FA/m={eval_res['false_alarms_per_minute']}", flush=True)

    # -------------------------------------------------------------
    # Aggregate Metrics Across Subjects
    # -------------------------------------------------------------
    final_comparison = {"cohort": active_cohort, "table": []}
    experiment_registry = {"metadata": {"cohort": active_cohort, "date": "2026-10-20"}, "experiments": {}}

    for exp_key, exp_data in exp_results.items():
        subs = exp_data["subjects"]
        tp_sum = sum(subs[s]["true_positives"] for s in subs)
        fp_sum = sum(subs[s]["false_positives"] for s in subs)
        fn_sum = sum(subs[s]["false_negatives"] for s in subs)
        gt_sum = sum(subs[s]["total_gt_events"] for s in subs)
        min_sum = sum(subs[s]["evaluated_minutes"] for s in subs)

        prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        rec = tp_sum / gt_sum if gt_sum else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        fa_rate = fp_sum / min_sum if min_sum else 0.0

        decision = "BASELINE"
        if exp_key == "EXP-1_DRIVER_ASSOCIATION":
            decision = "KEEP" if fp_sum < exp_results["EXP-0_V20_BASELINE"]["subjects"]["gC-14"]["false_positives"] + exp_results["EXP-0_V20_BASELINE"]["subjects"]["gZ-36"]["false_positives"] else "INCONCLUSIVE"
        elif exp_key == "EXP-2_ACTION_MODEL":
            prev_fp = final_comparison["table"][-1]["fp"]
            decision = "KEEP" if fp_sum <= prev_fp and f1 >= final_comparison["table"][-1]["f1"] else "INCONCLUSIVE"

        row = {
            "experiment": exp_key,
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "total_gt": gt_sum,
            "evaluated_minutes": round(min_sum, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_minute": round(fa_rate, 4),
            "decision": decision,
            "per_subject": {s: subs[s] for s in subs},
        }
        final_comparison["table"].append(row)
        experiment_registry["experiments"][exp_key] = row

    # -------------------------------------------------------------
    # Deliverables Generation
    # -------------------------------------------------------------
    print("\n=======================================================", flush=True)
    print("SAVING MACHINE-READABLE SCIENTIFIC ARTIFACTS...", flush=True)
    print("=======================================================", flush=True)

    (REPORTS_DIR / "experiment_registry.json").write_text(json.dumps(experiment_registry, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics_comparison.json").write_text(json.dumps(final_comparison, indent=2), encoding="utf-8")
    (REPORTS_DIR / "stage_rejection_ledger.json").write_text(json.dumps(stage_rejections[:500], indent=2), encoding="utf-8")

    # Full JSONL traces for EXP-2 (best candidate)
    with (REPORTS_DIR / "autonomous_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for r in all_raw_predictions:
            handle.write(json.dumps(r) + "\n")

    with (REPORTS_DIR / "phone_candidate_records.jsonl").open("w", encoding="utf-8") as handle:
        for r in all_candidate_records:
            handle.write(json.dumps(r) + "\n")

    with (REPORTS_DIR / "action_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for r in all_action_logs:
            handle.write(json.dumps(r) + "\n")

    with (REPORTS_DIR / "event_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for r in exp_results["EXP-2_ACTION_MODEL"]["predicted_events"]:
            handle.write(json.dumps(r) + "\n")

    event_matches_data = {
        "schema_version": "v21.0",
        "evaluation_type": "AUTONOMOUS_V21_UNIFIED_EVALUATION",
        "cohort": active_cohort,
        "experiments": final_comparison["table"],
    }
    (REPORTS_DIR / "event_matches.json").write_text(json.dumps(event_matches_data, indent=2), encoding="utf-8")

    print(json.dumps(final_comparison, indent=2), flush=True)
    return final_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Quick test run on 100 frames")
    parser.add_argument("--cohort", default="two-subject", choices=["two-subject", "three-subject"])
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()

    max_f = 100 if args.test_run else None
    run_unified_v21_experiments(cohort=args.cohort, sample_stride=args.stride, max_frames=max_f)
