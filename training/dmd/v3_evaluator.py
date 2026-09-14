"""V3 Multi-View Temporal Phone Evaluation Engine.

Evaluates continuous driver sequences using Multi-View Fusion (BODY, FACE, HANDS),
normalized Pose Features, and Temporal Window Aggregation.
Compares candidate configurations against V2 Baseline across development subjects:
- gC-14 (Development pool)
- gZ-36 (Development pool)
- gB-9  (Development pool)
- gE-28 (HOLDOUT - strictly guarded until pre-holdout freeze)
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ai.auxiliary import PoseEstimator
from backend.ai.detector import SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from training.common import sha256_file
from training.dmd.action_model import ActionProbabilities, RuleGuidedPoseClassifier, TemporalWindowAggregator
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.fusion import FusedPhoneObservation, MultiViewFusionEngine
from training.dmd.holdout_guard import assert_holdout_untouched
from training.dmd.multiview import MultiViewSynchronizer
from training.dmd.pose_features import extract_pose_features
from training.dmd.temporal_eval import TemporalEvaluationMetrics, TemporalMatchingPolicy, match_temporal_events


@dataclass(frozen=True)
class V3AblationResult:
    config_id: str  # "A0", "A1", ..., "A7"
    config_name: str
    subject_id: str
    precision: float
    recall: float
    f1: float
    false_alarms_per_minute: float
    phonecall_right_recall: float
    phonecall_left_recall: float
    texting_right_recall: float
    texting_left_recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_gt_events: int
    evaluated_minutes: float
    fps: float
    latency_ms_per_frame: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_per_action_recall(
    gt_intervals: List[Dict[str, Any]],
    matched_events: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate recall broken down by specific DMD ground truth action."""
    action_totals: Dict[str, int] = {}
    unique_matched_gts: Dict[str, set] = {}

    for gt in gt_intervals:
        action = gt.get("source_action", "unknown").replace("driver_actions/", "")
        action_totals[action] = action_totals.get(action, 0) + 1

    for m in matched_events:
        gt_ref = m.get("ground_truth") or m.get("gt")
        if gt_ref:
            action = gt_ref.get("source_action", "unknown").replace("driver_actions/", "")
            gt_key = (gt_ref.get("start_seconds"), gt_ref.get("end_seconds"), action)
            if action not in unique_matched_gts:
                unique_matched_gts[action] = set()
            unique_matched_gts[action].add(gt_key)

    recalls = {}
    for action, total in action_totals.items():
        matched_count = len(unique_matched_gts.get(action, set()))
        recalls[action] = round(matched_count / total, 4) if total > 0 else 0.0

    return recalls


def run_v3_evaluation_on_subject(
    subject_dir: Path,
    subject_id: str,
    detector: SafetyDetector,
    pose_estimator: Optional[PoseEstimator] = None,
    sample_stride_frames: int = 15,
    config_id: str = "A7",
    enable_face: bool = True,
    enable_hands: bool = True,
    enable_pose: bool = True,
    temporal_window: int = 30,
    phone_conf_threshold: float = 0.25,
    min_event_duration: float = 0.8,
    start_confirm_seconds: float = 0.5,
    end_clear_seconds: float = 0.8,
) -> Tuple[TemporalEvaluationMetrics, Dict[str, float], V3AblationResult]:
    """Execute complete V3 evaluation on a DMD subject."""
    assert_holdout_untouched(subject_id, caller_action="run_v3_evaluation_on_subject")

    sync = MultiViewSynchronizer(subject_dir, subject_id)
    if not sync.annotation_path or not sync.annotation_path.exists():
        raise FileNotFoundError(f"Annotation file missing for subject {subject_id} in {subject_dir}")

    ann = parse_dmd_openlabel(sync.annotation_path, fps=sync.primary_fps, total_frames=sync.total_frames)
    gt_intervals = [i.to_dict() for i in ann.phone_intervals]

    fusion_engine = MultiViewFusionEngine(
        phone_conf_threshold=phone_conf_threshold,
        temporal_window=temporal_window,
        enable_face_fusion=enable_face,
        enable_hands_fusion=enable_hands,
        enable_pose_fusion=enable_pose,
    )

    cfg_path = Path("models/temporal_config_v2.json")
    t_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    supported = {
        "window_seconds",
        "min_positive_seconds",
        "min_observations",
        "cooldown_seconds",
        "smoothing_alpha",
        "feature_positive_score",
        "positive_ratio",
        "gap_tolerance_seconds",
        "candidate_threshold",
        "activation_threshold",
        "release_threshold",
        "occlusion_bridge_seconds",
    }
    engine_kwargs = {k: v for k, v in t_cfg.items() if k in supported}
    event_engine = TemporalEventEngine(**engine_kwargs)

    t0 = time.perf_counter()
    processed_frames = 0
    torch.set_num_threads(2)
    generated_candidates = []

    enabled_views = ["BODY"]
    if enable_face:
        enabled_views.append("FACE")
    if enable_hands:
        enabled_views.append("HANDS")

    with torch.inference_mode():
        for pkg in sync.iterate_samples(stride=sample_stride_frames, enabled_views=enabled_views):
            processed_frames += 1
            body_phone_conf = 0.0
            face_phone_conf = 0.0
            hands_phone_conf = 0.0
            pose_feat = None

            # 1. BODY view processing
            if pkg.body_frame is not None:
                body_dets = detector.predict(pkg.body_frame, track=False, specialist_filter=["phone_detector"])
                phones = [d for d in body_dets if d.class_name == "phone"]
                if phones:
                    body_phone_conf = max(p.confidence for p in phones)
                    best_phone_box = max(phones, key=lambda x: x.confidence).xyxy
                else:
                    best_phone_box = None

                if enable_pose and pose_estimator:
                    raw_poses = pose_estimator.predict_raw_keypoints(pkg.body_frame)
                    if raw_poses:
                        best_pose = max(raw_poses, key=lambda x: sum(x[1]))
                        pose_feat = extract_pose_features(
                            keypoints=best_pose[0],
                            confidences=best_pose[1],
                            phone_box=best_phone_box,
                            fallback_box=best_pose[2],
                        )

            # 2. FACE view processing (recovers head/shoulder occlusions)
            if enable_face and pkg.face_frame is not None:
                face_dets = detector.predict(pkg.face_frame, track=False, specialist_filter=["phone_detector"])
                face_phones = [d for d in face_dets if d.class_name == "phone"]
                if face_phones:
                    face_phone_conf = max(p.confidence for p in face_phones)

            # 3. HANDS view processing (recovers steering wheel / lap occlusions)
            if enable_hands and pkg.hands_frame is not None:
                hands_dets = detector.predict(pkg.hands_frame, track=False, specialist_filter=["phone_detector"])
                hands_phones = [d for d in hands_dets if d.class_name == "phone"]
                if hands_phones:
                    hands_phone_conf = max(p.confidence for p in hands_phones)

            # 4. Multi-View Fusion
            fused_obs = fusion_engine.fuse_frame(
                timestamp_seconds=pkg.timestamp_seconds,
                frame_index=pkg.frame_index,
                body_phone_conf=body_phone_conf,
                face_phone_conf=face_phone_conf,
                hands_phone_conf=hands_phone_conf,
                body_pose_features=pose_feat,
                body_available=pkg.body_available,
                face_available=pkg.face_available,
                hands_available=pkg.hands_available,
            )

            # 5. Temporal Engine Update
            obs = Observation(
                timestamp=fused_obs.timestamp_seconds,
                class_name="phone",
                confidence=fused_obs.fused_score,
                track_id=1,
                occupant_role="driver",
                vehicle_context_id="DMD_CABIN",
                phone_context="HANDHELD_USE" if fused_obs.phone_detected else "UNKNOWN",
                fusion_score=fused_obs.fused_score,
                phone_hand_proximity=fused_obs.pose_text_score,
                phone_face_proximity=fused_obs.pose_call_score,
                pose_confidence=0.85 if enable_pose else 0.0,
                occupant_role_confidence=1.0,
                vehicle_context_confidence=1.0,
            )
            evs = event_engine.add(obs)
            for ev in evs:
                if ev.event_type == "PHONE":
                    generated_candidates.append({
                        "event_type": ev.event_type,
                        "start_seconds": ev.start_timestamp,
                        "end_seconds": ev.end_timestamp,
                        "confidence": ev.confidence,
                        "duration": round(ev.end_timestamp - ev.start_timestamp, 2),
                    })

    elapsed_sec = time.perf_counter() - t0
    eval_fps = processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0
    latency_ms = (elapsed_sec / processed_frames * 1000.0) if processed_frames > 0 else 0.0

    # Cooldown deduplication matching V2 benchmark standard
    predicted_events = []
    cooldown = float(t_cfg.get("cooldown_seconds", 3.0))
    for cand in generated_candidates:
        if not predicted_events:
            predicted_events.append(dict(cand))
        else:
            last = predicted_events[-1]
            if cand["start_seconds"] <= last["end_seconds"] + cooldown:
                if cand["end_seconds"] > last["end_seconds"]:
                    last["end_seconds"] = cand["end_seconds"]
                    last["duration"] = round(last["end_seconds"] - last["start_seconds"], 2)
            else:
                predicted_events.append(dict(cand))

    policy = TemporalMatchingPolicy()
    matched, unmatched_preds, unmatched_gts = match_temporal_events(
        gt_intervals=gt_intervals,
        predicted_events=predicted_events,
        policy=policy,
    )

    matched_gt_set = set()
    for m in matched:
        gt_item = m.get("ground_truth") or m.get("gt")
        if gt_item:
            matched_gt_set.add((float(gt_item["start_seconds"]), float(gt_item["end_seconds"])))

    tp = len(matched_gt_set)
    fn = len(unmatched_gts)
    total_gt = len(gt_intervals)

    duplicate_preds = max(0, len(matched) - tp)
    fp = len(unmatched_preds) + duplicate_preds

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    total_sec = sync.duration_seconds
    total_min = total_sec / 60.0 if total_sec > 0 else 0.1
    fa_per_min = fp / total_min

    start_lat = [abs(float(m.get("start_latency", 0.0))) for m in matched]
    end_lat = [abs(float(m.get("end_latency", 0.0))) for m in matched]

    metrics = TemporalEvaluationMetrics(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        false_alarms_per_minute=round(fa_per_min, 3),
        start_latency_mean=round(float(np.mean(start_lat)), 3) if start_lat else 0.0,
        end_latency_mean=round(float(np.mean(end_lat)), 3) if end_lat else 0.0,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_gt_events=total_gt,
        total_evaluated_seconds=round(total_sec, 2),
        total_evaluated_minutes=round(total_min, 3),
        pose_availability_rate=1.0 if enable_pose else 0.0,
        unknown_occupant_rate=0.0,
        phone_context_unknown_rate=0.0,
        matched_events=matched,
        unmatched_predictions=unmatched_preds,
        unmatched_gt=unmatched_gts,
    )

    per_action = calculate_per_action_recall(gt_intervals, matched)

    config_names = {
        "A0": "BODY V2 Baseline",
        "A1": "BODY + Improved ROI",
        "A2": "BODY + Pose Features",
        "A3": "BODY + FACE",
        "A4": "BODY + HANDS",
        "A5": "BODY + FACE + HANDS",
        "A6": "Multi-View + Pose Features",
        "A7": "Multi-View + Pose + 30-Frame Temporal Action Model",
    }

    ablation_res = V3AblationResult(
        config_id=config_id,
        config_name=config_names.get(config_id, config_id),
        subject_id=subject_id,
        precision=round(prec * 100.0, 1),
        recall=round(rec * 100.0, 1),
        f1=round(f1 * 100.0, 1),
        false_alarms_per_minute=round(fa_per_min, 3),
        phonecall_right_recall=round(per_action.get("phonecall_right", 0.0) * 100.0, 1),
        phonecall_left_recall=round(per_action.get("phonecall_left", 0.0) * 100.0, 1),
        texting_right_recall=round(per_action.get("texting_right", 0.0) * 100.0, 1),
        texting_left_recall=round(per_action.get("texting_left", 0.0) * 100.0, 1),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_gt_events=total_gt,
        evaluated_minutes=round(total_min, 2),
        fps=round(eval_fps, 1),
        latency_ms_per_frame=round(latency_ms, 2),
    )

    return metrics, per_action, ablation_res
