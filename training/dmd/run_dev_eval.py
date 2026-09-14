"""Quick evaluation runner for V3 development on gZ-36 and gC-14."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ai.auxiliary import PoseEstimator
from backend.ai.detector import SafetyDetector
from training.dmd.v3_evaluator import run_v3_evaluation_on_subject


def main():
    phone_model = Path("models/active/v2/phone_detector.pt")
    config_path = Path("models/model_config_v2.yaml")
    pose_model = Path("models/auxiliary/yolo11n-pose.pt")

    detector = SafetyDetector(phone_model, config_path=config_path)
    pose_estimator = PoseEstimator(pose_model)

    subj_dir = Path("datasets/external_dmd/subjects/gZ-36")
    print(f"Running A0 (BODY V2 Baseline) on gZ-36...")
    m_a0, acts_a0, res_a0 = run_v3_evaluation_on_subject(
        subject_dir=subj_dir,
        subject_id="gZ-36",
        detector=detector,
        pose_estimator=None,
        config_id="A0",
        enable_face=False,
        enable_hands=False,
        enable_pose=False,
        sample_stride_frames=15,
    )
    print(f"A0 Results on gZ-36:")
    print(f"  Precision: {res_a0.precision}%, Recall: {res_a0.recall}%, F1: {res_a0.f1}%, FA/min: {res_a0.false_alarms_per_minute}")
    print(f"  Actions: {acts_a0}")

    print(f"\nRunning A2 (BODY + Pose Features) on gZ-36...")
    m_a2, acts_a2, res_a2 = run_v3_evaluation_on_subject(
        subject_dir=subj_dir,
        subject_id="gZ-36",
        detector=detector,
        pose_estimator=pose_estimator,
        config_id="A2",
        enable_face=False,
        enable_hands=False,
        enable_pose=True,
        sample_stride_frames=15,
    )
    print(f"A2 Results on gZ-36:")
    print(f"  Precision: {res_a2.precision}%, Recall: {res_a2.recall}%, F1: {res_a2.f1}%, FA/min: {res_a2.false_alarms_per_minute}")
    print(f"  Actions: {acts_a2}")


if __name__ == "__main__":
    main()
