"""Feature Extractor for Roadwatch DMD Development Pool.

Extracts and caches multi-view detector and pose features for development subjects
(gC-14, gZ-36, gB-9). Strictly refuses consumed holdout subjects (gZ-37, gE-28).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ai.auxiliary import PoseEstimator
from backend.ai.detector import SafetyDetector
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.multiview import MultiViewSynchronizer
from training.dmd.pose_features import extract_pose_features

DERIVATIVES_DIR = Path("datasets/external_dmd/derivatives")


def extract_features_for_subject(
    subject_id: str,
    subject_dir: Path,
    stride: int = 15,
    force_recompute: bool = False,
) -> Path:
    assert_not_consumed_holdout_for_development(subject_id, caller_action="feature_extraction")
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DERIVATIVES_DIR / f"v3_features_{subject_id}.json"

    if out_path.exists() and not force_recompute:
        print(f"[CACHE] Found existing cached features for {subject_id} at {out_path}", flush=True)
        return out_path

    print(f"[EXTRACT] Starting feature extraction for {subject_id} at stride={stride}...", flush=True)
    phone_model_path = Path("models/active/v2/phone_detector.pt")
    config_path = Path("models/model_config_v2.yaml")
    pose_model_path = Path("models/auxiliary/yolo11n-pose.pt")

    detector = SafetyDetector(phone_model_path, config_path=config_path)
    pose_estimator = PoseEstimator(pose_model_path)

    sync = MultiViewSynchronizer(subject_dir, subject_id)
    has_face = any("rgb_face" in p.name.lower() for p in subject_dir.glob("*.mp4"))
    has_hands = any("rgb_hands" in p.name.lower() for p in subject_dir.glob("*.mp4"))

    enabled_views = ["BODY"]
    if has_face:
        enabled_views.append("FACE")
    if has_hands:
        enabled_views.append("HANDS")

    records: List[Dict] = []
    t0 = time.perf_counter()
    count = 0

    torch.set_num_threads(4)
    with torch.inference_mode():
        for pkg in sync.iterate_samples(stride=stride, enabled_views=enabled_views):
            count += 1
            body_phone_conf = 0.0
            face_phone_conf = 0.0
            hands_phone_conf = 0.0
            pose_arr: Optional[List[float]] = None

            # 1. BODY
            if pkg.body_frame is not None:
                body_dets = detector.predict(pkg.body_frame, track=False, specialist_filter=["phone_detector"])
                phones = [d for d in body_dets if d.class_name == "phone"]
                if phones:
                    body_phone_conf = max(p.confidence for p in phones)
                    best_phone_box = max(phones, key=lambda x: x.confidence).xyxy
                else:
                    best_phone_box = None

                raw_poses = pose_estimator.predict_raw_keypoints(pkg.body_frame)
                if raw_poses:
                    best_pose = max(raw_poses, key=lambda x: sum(x[1]))
                    pose_feat = extract_pose_features(
                        keypoints=best_pose[0],
                        confidences=best_pose[1],
                        phone_box=best_phone_box,
                        fallback_box=best_pose[2],
                    )
                    pose_arr = [round(float(x), 4) for x in pose_feat.to_array()]

            # 2. FACE
            if has_face and pkg.face_frame is not None:
                face_dets = detector.predict(pkg.face_frame, track=False, specialist_filter=["phone_detector"])
                face_phones = [d for d in face_dets if d.class_name == "phone"]
                if face_phones:
                    face_phone_conf = max(p.confidence for p in face_phones)

            # 3. HANDS
            if has_hands and pkg.hands_frame is not None:
                hands_dets = detector.predict(pkg.hands_frame, track=False, specialist_filter=["phone_detector"])
                hands_phones = [d for d in hands_dets if d.class_name == "phone"]
                if hands_phones:
                    hands_phone_conf = max(p.confidence for p in hands_phones)

            records.append({
                "frame_index": pkg.frame_index,
                "timestamp_seconds": round(float(pkg.timestamp_seconds), 4),
                "body_phone_conf": round(float(body_phone_conf), 4),
                "face_phone_conf": round(float(face_phone_conf), 4),
                "hands_phone_conf": round(float(hands_phone_conf), 4),
                "pose_features": pose_arr,
                "body_available": pkg.body_available,
                "face_available": pkg.face_available,
                "hands_available": pkg.hands_available,
            })

            if count % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{subject_id}] Processed {count} frames in {elapsed:.1f}s ({count/elapsed:.1f} FPS)...", flush=True)

    elapsed_total = time.perf_counter() - t0
    payload = {
        "subject_id": subject_id,
        "sample_stride": stride,
        "total_samples": len(records),
        "duration_seconds": sync.duration_seconds,
        "wall_time_seconds": round(elapsed_total, 2),
        "fps": round(len(records) / elapsed_total, 2) if elapsed_total > 0 else 0.0,
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[EXTRACT] Saved {len(records)} frames to {out_path} ({elapsed_total:.1f}s, {len(records)/elapsed_total:.1f} FPS)", flush=True)
    return out_path


if __name__ == "__main__":
    subjects = ["gB-9", "gZ-36", "gC-14"]
    for sid in subjects:
        sdir = Path("datasets/external_dmd/subjects") / sid
        if sdir.exists():
            extract_features_for_subject(sid, sdir, stride=15)
