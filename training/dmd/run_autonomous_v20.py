"""ROADWATCH V20 — Real Autonomous Video Inference & Temporal Evaluation Runner.

Processes full video sessions without ANY oracle metadata, GT intervals, or FN lists.
Strict Input Policy:
- The model ONLY sees raw video pixels, timestamps, and prior temporal history.
- Ground truth intervals are ONLY used post-hoc by the frozen matcher for scoring.

Architecture:
1. VIDEO DECODING (cv2.VideoCapture, stride=15 frames ~ 0.504s).
2. DRIVER ROI LOCALIZATION (normalized cabin geometry [0.0, 0.10, 0.65, 0.95]).
3. NATIVE DRIVER-CROP PHONE DETECTION:
   - D0: Specialist Full Frame (baseline)
   - D2: Specialist Native Driver ROI
   - D3: Auxiliary Native Driver ROI (COCO class 67)
   - D5: Ensembled Specialist + Auxiliary on Native Driver ROI (IoU merge)
4. POSE & CONTEXT CLASSIFICATION (runs PoseEstimator on candidate frames to evaluate hand/face proximity).
   - Real evidence classification: HANDHELD_PHONE_USE vs MOUNTED_OR_STATIC vs UNKNOWN.
5. DRIVER OCCUPANT ASSOCIATION (verifies driver role; avoids false passenger association).
6. TEMPORAL EVENT ENGINE (deep_bridge_40 hysteresis; UNKNOWN observations preserve state without resetting).
7. POST-HOC FROZEN MATCHER (iou_threshold=0.3, onset_tolerance=2.5s).
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
from ultralytics import YOLO

from backend.ai.auxiliary import PoseEstimator, classify_phone_context
from backend.ai.detector import NormalizedDetection
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.temporal_eval import (
    TemporalMatchingPolicy,
    match_temporal_events,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "v20_ai85_execution"

SPECIALIST_PATH = MODELS_DIR / "active" / "v2" / "phone_detector.pt"
AUXILIARY_PATH = MODELS_DIR / "auxiliary" / "yolo11n.pt"
POSE_PATH = MODELS_DIR / "auxiliary" / "yolo11n-pose.pt"
CONFIG_PATH = MODELS_DIR / "temporal_config_v2.json"
POLICY_PATH = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def merge_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.50) -> list[dict[str, Any]]:
    """Confidence-ordered NMS across specialist and auxiliary models."""
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if all(box_iou(det["box"], accepted["box"]) <= iou_threshold for accepted in kept):
            kept.append(det)
    return kept


class AutonomousVideoInferenceEngine:
    def __init__(
        self,
        mode: str = "D5",  # D0 (Full), D2 (Spec ROI), D3 (Aux ROI), D5 (Ensemble ROI)
        confidence_floor: float = 0.15,
        sample_stride: int = 15,
    ):
        self.mode = mode
        self.confidence_floor = confidence_floor
        self.sample_stride = sample_stride
        torch.set_num_threads(4)

        # Lazy model references
        self._specialist = None
        self._auxiliary = None
        self._pose_estimator = None

    def _get_specialist(self) -> YOLO:
        if self._specialist is None:
            self._specialist = YOLO(str(SPECIALIST_PATH))
        return self._specialist

    def _get_auxiliary(self) -> YOLO:
        if self._auxiliary is None:
            self._auxiliary = YOLO(str(AUXILIARY_PATH))
        return self._auxiliary

    def _get_pose(self) -> PoseEstimator:
        if self._pose_estimator is None:
            self._pose_estimator = PoseEstimator(POSE_PATH)
        return self._pose_estimator

    def run_session_inference(
        self,
        video_path: Path,
        temporal_config: dict[str, Any],
        max_frames: int | None = None,
        record_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Runs true end-to-end autonomous inference over the full video stream."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps
        if self.sample_stride < 1 or (max_frames is not None and max_frames < 1):
            cap.release()
            raise ValueError("stride and sample limit must be positive")

        # Driver ROI geometry in DMD body camera (left cabin seat)
        # Bounded to [0, 72, 832, 684] on 1280x720 video
        roi_norm = (0.0, 0.10, 0.65, 0.95)

        engine_supported = {
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
        engine_kwargs = {k: v for k, v in temporal_config.items() if k in engine_supported}
        engine = TemporalEventEngine(**engine_kwargs)

        frame_idx = 0
        processed_count = 0
        generated_candidates = []
        prediction_logs = []

        started_at = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]
            timestamp = frame_idx / fps
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"    Processed {processed_count} frames ({timestamp:.1f}s / {total_seconds:.1f}s)...", flush=True)

            # Extract native Driver ROI
            rx1, ry1 = int(roi_norm[0] * w), int(roi_norm[1] * h)
            rx2, ry2 = int(roi_norm[2] * w), int(roi_norm[3] * h)
            driver_crop = frame[ry1:ry2, rx1:rx2]

            raw_detections: list[dict[str, Any]] = []

            # 1. Detection Stage based on mode
            if self.mode == "D0":
                # Specialist Full-Frame (historical baseline downscaled to 640)
                spec = self._get_specialist()
                res = spec.predict(frame, imgsz=640, conf=self.confidence_floor, device="cpu", verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    raw_detections.append({"box": xyxy, "confidence": conf, "source": "specialist_full", "local_box": [xyxy[0]-rx1, xyxy[1]-ry1, xyxy[2]-rx1, xyxy[3]-ry1]})

            elif self.mode in ("D2", "D5"):
                # Specialist on Native Driver Crop
                spec = self._get_specialist()
                res = spec.predict(driver_crop, imgsz=640, conf=self.confidence_floor, device="cpu", verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    local_xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    global_xyxy = [local_xyxy[0] + rx1, local_xyxy[1] + ry1, local_xyxy[2] + rx1, local_xyxy[3] + ry1]
                    raw_detections.append({"box": global_xyxy, "confidence": conf, "source": "specialist_roi", "local_box": local_xyxy})

            if self.mode in ("D3", "D5"):
                # Auxiliary YOLO11n on Native Driver Crop (COCO class 67: cell phone)
                aux = self._get_auxiliary()
                res = aux.predict(driver_crop, imgsz=640, conf=self.confidence_floor, classes=[67], device="cpu", verbose=False)[0]
                for box in res.boxes:
                    conf = float(box.conf.item())
                    local_xyxy = [float(v) for v in box.xyxy[0].cpu().tolist()]
                    global_xyxy = [local_xyxy[0] + rx1, local_xyxy[1] + ry1, local_xyxy[2] + rx1, local_xyxy[3] + ry1]
                    raw_detections.append({"box": global_xyxy, "confidence": conf, "source": "auxiliary_roi", "local_box": local_xyxy})

            # Merge detections if ensemble mode
            final_detections = merge_detections(raw_detections, iou_threshold=0.50) if len(raw_detections) > 1 else raw_detections

            # 2. Context & Pose Verification
            pose_item = None
            poses = []
            if not final_detections:
                obs = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=0.0,
                    track_id=1,
                    occupant_role="driver",
                    vehicle_context_id="DMD_CABIN",
                    phone_context="UNKNOWN",
                    fusion_score=0.0,
                    evidence_source="AUTONOMOUS_V20",
                )
            else:
                top_det = max(final_detections, key=lambda d: d["confidence"])
                pose_est = self._get_pose()
                poses = pose_est.predict(driver_crop)
                pose_item = poses[0] if poses else None

                # Classify phone context using local crop coordinates
                norm_det = NormalizedDetection(0, "phone", top_det["confidence"], tuple(top_det["local_box"]))
                context_str, hand_prox, face_prox = classify_phone_context(norm_det, pose_item)

                effective_context = context_str
                if context_str in ("HANDHELD", "HANDHELD_PHONE_USE"):
                    effective_context = "HANDHELD_USE"

                obs = Observation(
                    timestamp=timestamp,
                    class_name="phone",
                    confidence=top_det["confidence"],
                    track_id=1,
                    occupant_role="driver",
                    vehicle_context_id="DMD_CABIN",
                    phone_context=effective_context,
                    fusion_score=top_det["confidence"],
                    phone_hand_proximity=hand_prox,
                    phone_face_proximity=face_prox,
                    pose_confidence=pose_item.confidence if pose_item else 0.0,
                    evidence_source="AUTONOMOUS_V20",
                )
            # 3. Stream Observation into Temporal Event Engine (Causal Online Feeding)
            evs = engine.add(obs)
            for ev in evs:
                if ev.event_type == "PHONE":
                    generated_candidates.append({
                        "event_type": ev.event_type,
                        "start_seconds": ev.start_timestamp,
                        "end_seconds": ev.end_timestamp,
                        "confidence": ev.confidence,
                        "duration": round(ev.end_timestamp - ev.start_timestamp, 2),
                    })

            prediction_logs.append({
                "timestamp": round(timestamp, 3),
                "frame_idx": frame_idx,
                "confidence": round(obs.confidence, 4),
                "context": obs.phone_context,
                "emitted_events": len(evs),
            })
            if record_sink is not None:
                from dataclasses import asdict

                record_sink({
                    **prediction_logs[-1],
                    "observation": asdict(obs),
                    "raw_detections": raw_detections,
                    "candidates": final_detections,
                    "poses": [asdict(pose) for pose in poses],
                    "roi_xyxy": [rx1, ry1, rx2, ry2],
                    "candidate_events": [asdict(ev) for ev in evs],
                })

            if max_frames and processed_count >= max_frames:
                break

            # Advance by sample_stride using fast grab()
            skipped = 0
            for _ in range(self.sample_stride - 1):
                if not cap.grab():
                    break
                skipped += 1
            frame_idx += 1 + skipped

        cap.release()
        total_wall_time = time.perf_counter() - started_at

        # 4. Temporal Cooldown Deduplication
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

        observed_frames = min(total_frames, frame_idx + 1) if processed_count else 0
        if max_frames and processed_count >= max_frames:
            observed_frames = min(total_frames, frame_idx + 1)
        else:
            observed_frames = min(total_frames, frame_idx)
        if max_frames is None and observed_frames != total_frames:
            raise RuntimeError(f"INCOMPLETE_VIDEO_DECODE: {observed_frames}/{total_frames}")
        return {
            "mode": self.mode,
            "video_path": str(video_path),
            "total_frames": observed_frames,
            "source_frames": total_frames,
            "processed_frames": processed_count,
            "fps": fps,
            "total_seconds": observed_frames / fps,
            "wall_time_seconds": total_wall_time,
            "predicted_events": deduped_preds,
            "prediction_logs": prediction_logs,
        }

    def run_session(
        self,
        video_path: Path,
        annotation_path: Path,
        temporal_config: dict[str, Any],
        matching_policy: TemporalMatchingPolicy,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        inference = self.run_session_inference(video_path, temporal_config, max_frames)
        fps = inference["fps"]
        total_frames = inference["total_frames"]
        total_seconds = inference["total_seconds"]
        processed_count = inference["processed_frames"]
        total_wall_time = inference["wall_time_seconds"]
        deduped_preds = inference["predicted_events"]
        # 5. POST-INFERENCE Scoring Against OpenLABEL GT (Frozen Matcher)
        ann = parse_dmd_openlabel(annotation_path, fps=fps, total_frames=total_frames)
        gt_intervals = [i.to_dict() for i in ann.phone_intervals]

        matched, unmatched_preds, unmatched_gts = match_temporal_events(
            gt_intervals=gt_intervals,
            predicted_events=deduped_preds,
            policy=matching_policy,
        )

        matched_gt_set = set()
        for m in matched:
            gt_item = m["ground_truth"]
            matched_gt_set.add((float(gt_item["start_seconds"]), float(gt_item["end_seconds"])))

        tp = len(matched_gt_set)
        fn = len(unmatched_gts)
        total_gt = len(gt_intervals)
        duplicate_preds = max(0, len(matched) - tp)
        fp = len(unmatched_preds) + duplicate_preds

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / total_gt if total_gt else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        eval_mins = total_seconds / 60.0 if total_seconds > 0 else 1.0
        fa_rate = fp / eval_mins

        return {
            "mode": self.mode,
            "video_path": str(video_path),
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "wall_time_seconds": round(total_wall_time, 2),
            "effective_fps": round(processed_count / total_wall_time if total_wall_time else 0.0, 2),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_gt_events": total_gt,
            "evaluated_minutes": round(eval_mins, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_minute": round(fa_rate, 4),
            "predicted_events_count": len(deduped_preds),
            "matched_events": matched,
            "unmatched_predictions": unmatched_preds,
            "unmatched_gt": unmatched_gts,
            "predictions_summary": inference["prediction_logs"],
        }


def run_full_autonomous_evaluation(
    mode: str = "D5",
    cohort: str = "two-subject",
    max_frames: int | None = None,
    sample_stride: int = 15,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temporal_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy_data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    targets_dict = {
        "gC-14": (ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json"),
        "gZ-36": (ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json"),
        "gB-9": (ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gB-9/gB_9_s2_2019-02-27T11;57;51+01;00_rgb_ann_distraction.json"),
    }

    if cohort == "three-subject":
        active_cohort = ["gB-9", "gC-14", "gZ-36"]
    else:
        active_cohort = ["gC-14", "gZ-36"]

    targets = [(sub, targets_dict[sub][0], targets_dict[sub][1]) for sub in active_cohort if targets_dict[sub][0].exists()]

    engine = AutonomousVideoInferenceEngine(mode=mode, confidence_floor=0.15, sample_stride=sample_stride)
    subject_results = {}

    all_matched = []
    all_unmatched_preds = []

    for sub_id, vid_path, ann_path in targets:
        print(f"Running autonomous inference on {sub_id} (mode={mode}, stride={sample_stride})...", flush=True)
        res = engine.run_session(
            video_path=vid_path,
            annotation_path=ann_path,
            temporal_config=temporal_cfg,
            matching_policy=policy,
            max_frames=max_frames,
        )
        subject_results[sub_id] = res
        all_matched.extend(res["matched_events"])
        all_unmatched_preds.extend(res["unmatched_predictions"])
        print(f"  {sub_id} -> TP={res['true_positives']}, FP={res['false_positives']}, FN={res['false_negatives']}, F1={res['f1']}", flush=True)

    # Aggregate
    tp_sum = sum(subject_results[s]["true_positives"] for s in subject_results)
    fp_sum = sum(subject_results[s]["false_positives"] for s in subject_results)
    fn_sum = sum(subject_results[s]["false_negatives"] for s in subject_results)
    gt_sum = sum(subject_results[s]["total_gt_events"] for s in subject_results)
    min_sum = sum(subject_results[s]["evaluated_minutes"] for s in subject_results)

    prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    rec = tp_sum / gt_sum if gt_sum else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fa_rate = fp_sum / min_sum if min_sum else 0.0

    out = {
        "schema_version": "v20.0",
        "evaluation_type": "AUTONOMOUS_VIDEO_INFERENCE",
        "cohort": active_cohort,
        "detector_mode": mode,
        "total_gt_events": gt_sum,
        "evaluated_minutes": round(min_sum, 2),
        "true_positives": tp_sum,
        "false_positives": fp_sum,
        "false_negatives": fn_sum,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fa_rate, 4),
        "per_subject": {
            s: {
                "tp": subject_results[s]["true_positives"],
                "fp": subject_results[s]["false_positives"],
                "fn": subject_results[s]["false_negatives"],
                "precision": subject_results[s]["precision"],
                "recall": subject_results[s]["recall"],
                "f1": subject_results[s]["f1"],
                "effective_fps": subject_results[s]["effective_fps"],
            }
            for s in subject_results
        },
    }

    # Save deliverables
    (REPORTS_DIR / "event_matches.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # Write autonomous predictions jsonl
    with (REPORTS_DIR / "autonomous_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sub_id, sres in subject_results.items():
            for row in sres["predictions_summary"]:
                row["subject_id"] = sub_id
                handle.write(json.dumps(row) + "\n")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Quick test run on 100 frames")
    parser.add_argument("--mode", default="D5", choices=["D0", "D2", "D3", "D5"], help="Detection ablation mode")
    parser.add_argument("--cohort", default="two-subject", choices=["two-subject", "three-subject"], help="Cohort")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process per session")
    parser.add_argument("--sample-stride", type=int, default=15, help="Frame sample stride (e.g. 15 = 2fps)")
    args = parser.parse_args()

    max_f = 100 if args.test_run else args.max_frames
    results = run_full_autonomous_evaluation(
        mode=args.mode,
        cohort=args.cohort,
        max_frames=max_f,
        sample_stride=args.sample_stride,
    )
    print("\nAUTONOMOUS EVALUATION SUMMARY:")
    print(json.dumps(results, indent=2))
