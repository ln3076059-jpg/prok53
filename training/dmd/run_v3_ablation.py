"""Automated V3 Ablation Matrix Runner for DMD Subjects.

Executes ablation matrix configurations:
- A0: BODY V2 Baseline (Center-cabin camera only, single-frame detection + temporal window)
- A2: BODY + Pose Features (Center-cabin + 16D geometric pose features)
- A3: BODY + FACE (Multi-view: recovers left-ear head/shoulder occlusions)
- A4: BODY + HANDS (Multi-view: recovers lap/steering wheel texting occlusions)
- A5: BODY + FACE + HANDS (Full multi-view visual evidence)
- A6: Multi-View + Pose (Full multi-view visual + geometric pose features)
- A7: Multi-View + Pose + 30-Frame Temporal Window (Full V3 pipeline)

Strictly enforces Holdout Governance: gE-28 is guarded until Pre-Holdout Freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ai.auxiliary import PoseEstimator
from backend.ai.detector import SafetyDetector
from training.common import sha256_file
from training.dmd.holdout_guard import assert_holdout_untouched
from training.dmd.v3_evaluator import V3AblationResult, run_v3_evaluation_on_subject

CONFIG_SPECS = {
    "A0": {
        "name": "BODY V2 Baseline",
        "enable_face": False,
        "enable_hands": False,
        "enable_pose": False,
        "temporal_window": 15,
        "requires_face": False,
        "requires_hands": False,
    },
    "A2": {
        "name": "BODY + Pose Features",
        "enable_face": False,
        "enable_hands": False,
        "enable_pose": True,
        "temporal_window": 15,
        "requires_face": False,
        "requires_hands": False,
    },
    "A3": {
        "name": "BODY + FACE",
        "enable_face": True,
        "enable_hands": False,
        "enable_pose": False,
        "temporal_window": 15,
        "requires_face": True,
        "requires_hands": False,
    },
    "A4": {
        "name": "BODY + HANDS",
        "enable_face": False,
        "enable_hands": True,
        "enable_pose": False,
        "temporal_window": 15,
        "requires_face": False,
        "requires_hands": True,
    },
    "A5": {
        "name": "BODY + FACE + HANDS",
        "enable_face": True,
        "enable_hands": True,
        "enable_pose": False,
        "temporal_window": 15,
        "requires_face": True,
        "requires_hands": True,
    },
    "A6": {
        "name": "Multi-View + Pose Features",
        "enable_face": True,
        "enable_hands": True,
        "enable_pose": True,
        "temporal_window": 15,
        "requires_face": True,
        "requires_hands": True,
    },
    "A7": {
        "name": "Multi-View + Pose + 30-Frame Temporal Window",
        "enable_face": True,
        "enable_hands": True,
        "enable_pose": True,
        "temporal_window": 30,
        "requires_face": True,
        "requires_hands": True,
    },
}


def run_subject_ablations(
    subject_id: str,
    subject_dir: Path,
    configs_to_run: List[str],
    sample_stride: int = 15,
    output_json: Path | None = None,
) -> List[Dict[str, Any]]:
    # 1. Enforce holdout guard
    assert_holdout_untouched(subject_id, caller_action=f"run_subject_ablations({subject_id})")

    # 2. Initialize active models
    phone_model_path = Path("models/active/v2/phone_detector.pt")
    config_path = Path("models/model_config_v2.yaml")
    pose_model_path = Path("models/auxiliary/yolo11n-pose.pt")

    if not phone_model_path.exists():
        raise FileNotFoundError(f"Missing phone model at {phone_model_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Missing pose model at {pose_model_path}")

    print("=================================================================", flush=True)
    print(f"STARTING V3 ABLATION RUN FOR SUBJECT: {subject_id}", flush=True)
    print(f"Subject directory: {subject_dir}", flush=True)
    print(f"Configurations: {configs_to_run}", flush=True)
    print(f"Sample stride: {sample_stride} frames (~2.0 fps at 29.76 fps)", flush=True)
    print("=================================================================", flush=True)

    print("[INIT] Loading SafetyDetector...", flush=True)
    detector = SafetyDetector(phone_model_path, config_path=config_path)
    print("[INIT] Loading PoseEstimator...", flush=True)
    pose_estimator = PoseEstimator(pose_model_path)

    # Check available streams
    has_face = any("rgb_face" in p.name.lower() for p in subject_dir.glob("*.mp4"))
    has_hands = any("rgb_hands" in p.name.lower() for p in subject_dir.glob("*.mp4"))
    print(f"[STREAMS] Subject {subject_id} available views: BODY=True, FACE={has_face}, HANDS={has_hands}", flush=True)

    results = []

    for cid in configs_to_run:
        if cid not in CONFIG_SPECS:
            print(f"[WARN] Unknown config ID: {cid}, skipping.", flush=True)
            continue

        spec = CONFIG_SPECS[cid]
        if spec["requires_face"] and not has_face:
            print(f"[SKIP] Config {cid} requires FACE stream which is not available for {subject_id}.", flush=True)
            continue
        if spec["requires_hands"] and not has_hands:
            print(f"[SKIP] Config {cid} requires HANDS stream which is not available for {subject_id}.", flush=True)
            continue

        print(f"\n---> Executing Config {cid}: {spec['name']} ...", flush=True)
        t_start = time.perf_counter()

        metrics, per_action, ablation_res = run_v3_evaluation_on_subject(
            subject_dir=subject_dir,
            subject_id=subject_id,
            detector=detector,
            pose_estimator=pose_estimator if spec["enable_pose"] else None,
            sample_stride_frames=sample_stride,
            config_id=cid,
            enable_face=spec["enable_face"],
            enable_hands=spec["enable_hands"],
            enable_pose=spec["enable_pose"],
            temporal_window=spec["temporal_window"],
        )
        elapsed = time.perf_counter() - t_start

        res_dict = ablation_res.to_dict()
        res_dict["per_action_recall_breakdown"] = per_action
        res_dict["execution_wall_seconds"] = round(elapsed, 2)
        results.append(res_dict)

        print(f"     Results for {cid} ({spec['name']}):", flush=True)
        print(f"       Precision: {ablation_res.precision}% | Recall: {ablation_res.recall}% | F1: {ablation_res.f1}%", flush=True)
        print(f"       False Alarms/min: {ablation_res.false_alarms_per_minute} (TP={ablation_res.true_positives}, FP={ablation_res.false_positives}, FN={ablation_res.false_negatives})", flush=True)
        print(f"       Action Breakdown: call_right={ablation_res.phonecall_right_recall}%, call_left={ablation_res.phonecall_left_recall}%, text_right={ablation_res.texting_right_recall}%, text_left={ablation_res.texting_left_recall}%", flush=True)
        print(f"       Wall time: {elapsed:.1f}s | FPS: {ablation_res.fps} | Latency: {ablation_res.latency_ms_per_frame} ms/frame", flush=True)

    # Summary table
    print("\n" + "=" * 80, flush=True)
    print(f"ABLATION SUMMARY FOR SUBJECT: {subject_id}", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Config':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'FA/min':<8} | {'Call-R':<8} | {'Call-L':<8} | {'Text-R':<8} | {'Text-L':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        row = (
            f"{r['config_id']:<8} | "
            f"{r['precision']:>7.1f}% | "
            f"{r['recall']:>6.1f}% | "
            f"{r['f1']:>5.1f} | "
            f"{r['false_alarms_per_minute']:>8.3f} | "
            f"{r['phonecall_right_recall']:>7.1f}% | "
            f"{r['phonecall_left_recall']:>7.1f}% | "
            f"{r['texting_right_recall']:>7.1f}% | "
            f"{r['texting_left_recall']:>7.1f}%"
        )
        print(row, flush=True)
    print("=" * 80, flush=True)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subject_id": subject_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "sample_stride_frames": sample_stride,
            "results": results,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVED] Results saved to {output_json}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run V3 ablation matrix on a DMD subject.")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. gZ-36, gB-9, gC-14)")
    parser.add_argument("--configs", nargs="+", default=["A0", "A2", "A3", "A4", "A5", "A6", "A7"], help="Config IDs to run")
    parser.add_argument("--stride", type=int, default=15, help="Sampling stride in frames (default: 15)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    subj_dir = Path(f"datasets/external_dmd/subjects/{args.subject}")
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory does not exist: {subj_dir}")

    out_path = Path(args.output) if args.output else Path(f"reports/ablation_v3_{args.subject}.json")
    run_subject_ablations(
        subject_id=args.subject,
        subject_dir=subj_dir,
        configs_to_run=args.configs,
        sample_stride=args.stride,
        output_json=out_path,
    )


if __name__ == "__main__":
    main()
