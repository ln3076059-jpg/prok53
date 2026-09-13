"""DMD Temporal Phone Calibration Runner.

Performs parameter sweep across development calibration subjects only.
Extracts sequence observations once and evaluates candidate configurations in memory,
recording reproducible calibration artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.ai.auxiliary import PoseEstimator
from backend.ai.detector import SafetyDetector
from training.common import sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.temporal_eval import (
    TemporalEvaluationMetrics,
    TemporalMatchingPolicy,
    evaluate_temporal_engine_on_observations,
    extract_sequence_observations,
)


CANDIDATE_CONFIGS = [
    # 1.5s window baseline (~45 video frames at 30 fps)
    {
        "name": "window_15f_fast",
        "window_seconds": 1.50,
        "min_positive_seconds": 0.50,
        "min_observations": 2,
        "positive_ratio": 0.50,
        "cooldown_seconds": 3.0,
        "gap_tolerance_seconds": 1.50,
        "feature_positive_score": 0.28,
        "candidate_threshold": 0.28,
        "activation_threshold": 0.32,
        "release_threshold": 0.20,
    },
    # 2.0s window balanced (~60 video frames at 30 fps)
    {
        "name": "window_20f_balanced",
        "window_seconds": 2.00,
        "min_positive_seconds": 0.60,
        "min_observations": 2,
        "positive_ratio": 0.50,
        "cooldown_seconds": 4.0,
        "gap_tolerance_seconds": 1.50,
        "feature_positive_score": 0.28,
        "candidate_threshold": 0.30,
        "activation_threshold": 0.35,
        "release_threshold": 0.20,
    },
    # 2.5s window conservative (~75 video frames at 30 fps)
    {
        "name": "window_30f_conservative",
        "window_seconds": 2.50,
        "min_positive_seconds": 0.80,
        "min_observations": 3,
        "positive_ratio": 0.55,
        "cooldown_seconds": 4.0,
        "gap_tolerance_seconds": 1.50,
        "feature_positive_score": 0.28,
        "candidate_threshold": 0.32,
        "activation_threshold": 0.38,
        "release_threshold": 0.20,
    },
    # 3.0s window strict (~90 video frames at 30 fps)
    {
        "name": "window_45f_strict",
        "window_seconds": 3.00,
        "min_positive_seconds": 1.00,
        "min_observations": 3,
        "positive_ratio": 0.60,
        "cooldown_seconds": 5.0,
        "gap_tolerance_seconds": 1.50,
        "feature_positive_score": 0.28,
        "candidate_threshold": 0.35,
        "activation_threshold": 0.40,
        "release_threshold": 0.20,
    },
]


def run_calibration_sweep(
    calibration_items: list[dict[str, Any]],
    phone_model_path: Path,
    pose_model_path: Path | None = None,
    output_json_path: Path = Path("reports/DMD_PHONE_TEMPORAL_CALIBRATION.json"),
    output_md_path: Path = Path("reports/DMD_PHONE_TEMPORAL_CALIBRATION.md"),
    stride: int = 15,
) -> dict[str, Any]:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    detector = SafetyDetector(phone_model_path, config_path=Path("models/model_config_v2.yaml"))
    pose_estimator = PoseEstimator(pose_model_path) if (pose_model_path and pose_model_path.exists()) else None

    model_hash = sha256_file(phone_model_path)
    cal_subjects = sorted({str(i["source_participant_id"]) for i in calibration_items})

    # Step 1: Pre-extract observations for each calibration video once
    video_data = []
    print(f"Pre-extracting frame observations for {len(calibration_items)} calibration video(s)...")
    for item in calibration_items:
        vid = Path(item["member_path"])
        ann_path = Path(item["annotation_member_path"])
        print(f"  Extracting observations from: {vid.name} (stride={stride})...")
        obs, diag = extract_sequence_observations(
            video_path=vid,
            detector=detector,
            pose_estimator=pose_estimator,
            sample_stride_frames=stride,
        )
        ann = parse_dmd_openlabel(ann_path, fps=diag["fps"], total_frames=diag["total_frames"])
        gt_list = [i.to_dict() for i in ann.phone_intervals]
        video_data.append({
            "observations": obs,
            "diagnostics": diag,
            "gt_list": gt_list,
        })
        print(f"  Extracted {len(obs)} observations, {len(gt_list)} GT intervals.")

    # Step 2: In-memory evaluation over candidate configurations
    results_by_config = []
    best_config = None
    best_f1 = -1.0

    print(f"Evaluating {len(CANDIDATE_CONFIGS)} candidate configurations against cached observations...")
    for cfg in CANDIDATE_CONFIGS:
        cfg_name = cfg["name"]
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_mins = 0.0
        start_lats = []
        end_lats = []
        total_gt = 0

        for v in video_data:
            metrics = evaluate_temporal_engine_on_observations(
                observations=v["observations"],
                gt_intervals=v["gt_list"],
                diagnostics=v["diagnostics"],
                temporal_config=cfg,
            )
            total_tp += metrics.true_positives
            total_fp += metrics.false_positives
            total_fn += metrics.false_negatives
            total_gt += metrics.total_gt_events
            total_mins += metrics.total_evaluated_minutes
            start_lats.append(metrics.start_latency_mean)
            end_lats.append(metrics.end_latency_mean)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fa_per_min = total_fp / max(total_mins, 0.1)

        summary = {
            "config_name": cfg_name,
            "parameters": cfg,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_minute": round(fa_per_min, 3),
            "start_latency_mean": round(sum(start_lats) / len(start_lats) if start_lats else 0.0, 3),
            "end_latency_mean": round(sum(end_lats) / len(end_lats) if end_lats else 0.0, 3),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "total_gt_events": total_gt,
            "total_evaluated_minutes": round(total_mins, 2),
        }
        results_by_config.append(summary)

        if fa_per_min <= 1.0 and f1 > best_f1:
            best_f1 = f1
            best_config = cfg
        elif best_config is None and f1 > best_f1:
            best_f1 = f1
            best_config = cfg

    if best_config is None:
        best_config = CANDIDATE_CONFIGS[1]

    config_json_str = json.dumps(best_config, sort_keys=True)
    config_hash = hashlib.sha256(config_json_str.encode("utf-8")).hexdigest()

    output_data = {
        "status": "COMPLETE",
        "dataset_role": "TEMPORAL_DEVELOPMENT",
        "calibration_subjects": cal_subjects,
        "phone_model_path": str(phone_model_path.as_posix()),
        "phone_model_sha256": model_hash,
        "selected_configuration": best_config,
        "selected_configuration_hash": config_hash,
        "selection_criterion": "Highest F1 with false_alarms_per_minute <= 1.0 on calibration set",
        "results_by_config": results_by_config,
    }

    output_json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Saved calibration JSON to {output_json_path}")

    # Generate Markdown report
    md_content = f"""# DMD Phone Temporal Calibration Report

**Status:** COMPLETE  
**Dataset Role:** TEMPORAL_DEVELOPMENT (Non-Canonical, Development Only)  
**Calibration Subjects:** {', '.join(cal_subjects)}  
**Phone Model SHA256:** `{model_hash}`  
**Selected Config Hash:** `{config_hash}`  
**Selection Criterion:** Highest F1 with false alarms / min $\\le 1.0$  

---

## 1. Candidate Comparison Grid

| Configuration | Window (s) | Min Pos (s) | Pos Ratio | Precision | Recall | F1 | False Alarms / min | Start Latency (s) |
|---|---|---|---|---|---|---|---|---|
"""
    for r in results_by_config:
        p = r["parameters"]
        sel = " **(SELECTED)**" if p["name"] == best_config["name"] else ""
        md_content += (
            f"| `{p['name']}`{sel} | {p['window_seconds']:.2f} | {p['min_positive_seconds']:.2f} | "
            f"{p['positive_ratio']:.2f} | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | "
            f"{r['f1']*100:.1f}% | {r['false_alarms_per_minute']:.2f} | {r['start_latency_mean']:+.2f}s |\n"
        )

    md_content += f"""
---

## 2. Selected Frozen Calibration Parameters

```json
{json.dumps(best_config, indent=2)}
```

This configuration is permanently frozen for the held-subject benchmark.
"""
    output_md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved calibration Markdown report to {output_md_path}")

    return output_data


def main() -> None:
    manifest_path = Path("datasets/external_dmd/manifests/dmd_sources.jsonl")
    if not manifest_path.exists():
        print(f"Manifest {manifest_path} not found.")
        return
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # Calibration items are subjects != 36 (e.g., 14, 37)
    cal_items = [x for x in items if str(x.get("source_participant_id")) != "36"]
    if not cal_items:
        print("No calibration items found.")
        return

    run_calibration_sweep(
        calibration_items=cal_items,
        phone_model_path=Path("models/active/v2/phone_detector.pt"),
        pose_model_path=Path("models/auxiliary/yolo11n-pose.pt"),
    )


if __name__ == "__main__":
    main()

