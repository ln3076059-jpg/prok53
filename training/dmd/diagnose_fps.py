"""Diagnostic script to run V20 and collect all 16 FPs and 14 FNs with exact root causes."""
from __future__ import annotations

import json
from pathlib import Path
from ultralytics import YOLO

from backend.ai.auxiliary import PoseEstimator, classify_phone_context
from backend.ai.detector import NormalizedDetection
from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.temporal_eval import (
    TemporalMatchingPolicy,
    match_temporal_events,
)
from training.dmd.run_autonomous_v20 import AutonomousVideoInferenceEngine

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CONFIG_PATH = MODELS_DIR / "temporal_config_v2.json"
POLICY_PATH = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"

def main():
    temporal_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy_data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    targets = [
        ("gC-14", ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json"),
        ("gZ-36", ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4", ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json"),
    ]

    engine = AutonomousVideoInferenceEngine(mode="D5", confidence_floor=0.15, sample_stride=15)
    
    diagnostic_output = {}

    for sub_id, vid_path, ann_path in targets:
        print(f"Running full session on {sub_id}...", flush=True)
        res = engine.run_session(vid_path, ann_path, temporal_cfg, policy)
        diagnostic_output[sub_id] = {
            "tp": res["true_positives"],
            "fp": res["false_positives"],
            "fn": res["false_negatives"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "matched_events": res["matched_events"],
            "unmatched_predictions": res["unmatched_predictions"],
            "unmatched_gt": res["unmatched_gt"],
        }
        print(f"Done {sub_id}: TP={res['true_positives']}, FP={res['false_positives']}, FN={res['false_negatives']}, F1={res['f1']}")

    out_file = ROOT / "reports" / "v20_ai85_execution" / "v20_raw_diagnostic_ledger.json"
    out_file.write_text(json.dumps(diagnostic_output, indent=2), encoding="utf-8")
    print(f"Saved complete diagnostic ledger to {out_file}")

if __name__ == "__main__":
    main()
