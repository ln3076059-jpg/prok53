"""DMD Temporal Phone Evaluation and Calibration Engine.

Evaluates continuous driver sequences using active V2 models,
extracts frame-level phone and pose observations, feeds them into
TemporalEventEngine with parameterized hysteresis, and matches generated
events against official OpenLABEL ground truth.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml
import torch
import gc

from backend.ai.association import OccupantAssociator
from backend.ai.auxiliary import PoseEstimator, classify_phone_context
from backend.ai.cabin import CabinLocalizer
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine
from training.common import sha256_file
from training.dmd.adapter import DMDSessionAnnotation, parse_dmd_openlabel


@dataclass(frozen=True)
class TemporalMatchingPolicy:
    iou_threshold: float = 0.3
    onset_tolerance_seconds: float = 2.5
    offset_tolerance_seconds: float = 3.5


@dataclass(frozen=True)
class TemporalEvaluationMetrics:
    precision: float
    recall: float
    f1: float
    false_alarms_per_minute: float
    start_latency_mean: float
    end_latency_mean: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_gt_events: int
    total_evaluated_seconds: float
    total_evaluated_minutes: float
    pose_availability_rate: float
    unknown_occupant_rate: float
    phone_context_unknown_rate: float
    matched_events: list[dict[str, Any]]
    unmatched_predictions: list[dict[str, Any]]
    unmatched_gt: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_temporal_events(
    gt_intervals: list[dict[str, Any]],
    predicted_events: list[dict[str, Any]],
    policy: TemporalMatchingPolicy = TemporalMatchingPolicy(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Matches predicted events against ground truth intervals using IoU or onset tolerance."""
    matched = []
    unmatched_preds = list(predicted_events)
    unmatched_gts = list(gt_intervals)

    gt_matched_flags = [False] * len(gt_intervals)

    for pred in predicted_events:
        p_start = float(pred["start_seconds"])
        p_end = float(pred["end_seconds"])
        best_gt_idx = -1
        best_overlap = 0.0

        for idx, gt in enumerate(gt_intervals):
            g_start = float(gt["start_seconds"])
            g_end = float(gt["end_seconds"])

            inter = max(0.0, min(p_end, g_end) - max(p_start, g_start))
            union = max(p_end, g_end) - min(p_start, g_start)
            iou = inter / union if union > 0 else 0.0

            onset_diff = abs(p_start - g_start)
            pred_dur = max(0.1, p_end - p_start)
            pred_overlap = inter / pred_dur

            is_match = (
                (iou >= policy.iou_threshold)
                or (onset_diff <= policy.onset_tolerance_seconds and inter > 0.5)
                or (pred_overlap >= 0.40 and inter >= 0.8)
            )

            if is_match and (iou > best_overlap or best_gt_idx < 0):
                best_overlap = iou
                best_gt_idx = idx

        if best_gt_idx >= 0:
            gt = gt_intervals[best_gt_idx]
            match_record = {
                "prediction": pred,
                "ground_truth": gt,
                "iou": round(best_overlap, 3),
                "start_latency": round(p_start - float(gt["start_seconds"]), 3),
                "end_latency": round(p_end - float(gt["end_seconds"]), 3),
            }
            matched.append(match_record)
            if not gt_matched_flags[best_gt_idx]:
                gt_matched_flags[best_gt_idx] = True
                if gt in unmatched_gts:
                    unmatched_gts.remove(gt)
            if pred in unmatched_preds:
                unmatched_preds.remove(pred)

    return matched, unmatched_preds, unmatched_gts


def extract_sequence_observations(
    video_path: Path,
    detector: SafetyDetector,
    pose_estimator: PoseEstimator | None = None,
    sample_stride_frames: int = 15,
) -> tuple[list[Observation], dict[str, Any]]:
    """Extract frame-level observations and diagnostics once per video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps if fps > 0 else 0.0

    frame_idx = 0
    pose_seen = 0
    role_unknown = 0
    context_unknown = 0
    total_processed_frames = 0
    observations: list[Observation] = []

    with torch.inference_mode():
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp = frame_idx / fps
            total_processed_frames += 1

            detections = detector.predict(frame, track=False)
            phone_dets = [d for d in detections if d.class_name == "phone"]

            pose_res = pose_estimator.predict(frame) if pose_estimator else []
            pose_item = pose_res[0] if pose_res else None
            if pose_item is not None:
                pose_seen += 1

            if not phone_dets:
                empty_obs = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=0.0,
                    track_id=1,
                    occupant_role="driver",
                    vehicle_context_id="DMD_CABIN",
                    phone_context="UNKNOWN",
                    fusion_score=0.0,
                    phone_hand_proximity=0.0,
                    phone_face_proximity=0.0,
                    pose_confidence=0.0,
                    occupant_role_confidence=1.0,
                    vehicle_context_confidence=1.0,
                )
                observations.append(empty_obs)
            else:
                for d in phone_dets:
                    role = "driver"
                    role_conf = 0.95
                    pose_conf = pose_item.confidence if pose_item else 0.0

                    context, hand_prox, face_prox = classify_phone_context(
                        detection=d,
                        pose=pose_item,
                    )
                    if context in ("UNKNOWN", "UNKNOWN_PHONE_CONTEXT"):
                        context_unknown += 1

                    obs = Observation(
                        timestamp=timestamp,
                        class_name="phone",
                        confidence=d.confidence,
                        track_id=1,
                        occupant_role=role,
                        vehicle_context_id="DMD_CABIN",
                        phone_context=context,
                        fusion_score=d.confidence,
                        phone_hand_proximity=hand_prox,
                        phone_face_proximity=face_prox,
                        pose_confidence=pose_conf,
                        occupant_role_confidence=role_conf,
                        vehicle_context_confidence=1.0,
                    )
                    observations.append(obs)

            if total_processed_frames % 50 == 0:
                gc.collect()

            frame_idx += sample_stride_frames

    cap.release()

    diagnostics = {
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "fps": fps,
        "processed_frames": total_processed_frames,
        "pose_seen": pose_seen,
        "role_unknown": role_unknown,
        "context_unknown": context_unknown,
    }
    return observations, diagnostics


def evaluate_temporal_engine_on_observations(
    observations: list[Observation],
    gt_intervals: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    temporal_config: dict[str, Any],
    matching_policy: TemporalMatchingPolicy = TemporalMatchingPolicy(),
) -> TemporalEvaluationMetrics:
    """Evaluates a temporal engine configuration on pre-extracted observations in milliseconds."""
    # Filter config to only pass supported args
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
    }
    engine_kwargs = {k: v for k, v in temporal_config.items() if k in supported}
    engine = TemporalEventEngine(**engine_kwargs)

    generated_candidates = []
    for obs in observations:
        events = engine.add(obs)
        for ev in events:
            if ev.event_type == "PHONE":
                generated_candidates.append({
                    "event_type": ev.event_type,
                    "start_seconds": ev.start_timestamp,
                    "end_seconds": ev.end_timestamp,
                    "confidence": ev.confidence,
                    "duration": round(ev.end_timestamp - ev.start_timestamp, 2),
                })

    deduped_preds = []
    cooldown = float(temporal_config.get("cooldown_seconds", 3.0))
    for cand in generated_candidates:
        if not deduped_preds:
            deduped_preds.append(dict(cand))
        else:
            last = deduped_preds[-1]
            if cand["start_seconds"] <= last["end_seconds"] + cooldown:
                if cand["end_seconds"] > last["end_seconds"]:
                    last["end_seconds"] = cand["end_seconds"]
                    last["duration"] = round(last["end_seconds"] - last["start_seconds"], 2)
            else:
                deduped_preds.append(dict(cand))

    matched, unmatched_preds, unmatched_gts = match_temporal_events(
        gt_intervals=gt_intervals,
        predicted_events=deduped_preds,
        policy=matching_policy,
    )

    tp = len(matched)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    total_gt = len(gt_intervals)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = (total_gt - fn) / total_gt if total_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    total_seconds = diagnostics["total_seconds"]
    eval_mins = total_seconds / 60.0 if total_seconds > 0 else 1.0
    fa_per_min = fp / eval_mins

    start_lats = [m["start_latency"] for m in matched]
    end_lats = [m["end_latency"] for m in matched]

    mean_start_lat = statistics.mean(start_lats) if start_lats else 0.0
    mean_end_lat = statistics.mean(end_lats) if end_lats else 0.0

    processed_frames = max(diagnostics["processed_frames"], 1)
    pose_avail_rate = diagnostics["pose_seen"] / processed_frames
    unk_occ_rate = diagnostics["role_unknown"] / processed_frames
    unk_ctx_rate = diagnostics["context_unknown"] / processed_frames

    return TemporalEvaluationMetrics(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        false_alarms_per_minute=round(fa_per_min, 3),
        start_latency_mean=round(mean_start_lat, 3),
        end_latency_mean=round(mean_end_lat, 3),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_gt_events=len(gt_intervals),
        total_evaluated_seconds=round(total_seconds, 2),
        total_evaluated_minutes=round(eval_mins, 2),
        pose_availability_rate=round(pose_avail_rate, 3),
        unknown_occupant_rate=round(unk_occ_rate, 3),
        phone_context_unknown_rate=round(unk_ctx_rate, 3),
        matched_events=matched,
        unmatched_predictions=unmatched_preds,
        unmatched_gt=unmatched_gts,
    )


def run_sequence_phone_evaluation(
    video_path: Path,
    annotation_path: Path,
    temporal_config: dict[str, Any],
    detector: SafetyDetector,
    pose_estimator: PoseEstimator | None = None,
    sample_stride_frames: int = 15,
    matching_policy: TemporalMatchingPolicy = TemporalMatchingPolicy(),
) -> TemporalEvaluationMetrics:
    observations, diagnostics = extract_sequence_observations(
        video_path=video_path,
        detector=detector,
        pose_estimator=pose_estimator,
        sample_stride_frames=sample_stride_frames,
    )
    ann = parse_dmd_openlabel(annotation_path, fps=diagnostics["fps"], total_frames=diagnostics["total_frames"])
    gt_list = [i.to_dict() for i in ann.phone_intervals]

    return evaluate_temporal_engine_on_observations(
        observations=observations,
        gt_intervals=gt_list,
        diagnostics=diagnostics,
        temporal_config=temporal_config,
        matching_policy=matching_policy,
    )
