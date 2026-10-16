"""Governed Roadwatch V4 development ablations.

Only the declared development subjects are accepted. Consumed holdouts are rejected by the shared
guard, and baseline/candidate configurations reuse the existing DMD annotations, cached detector
features, temporal evaluator, and event-matching policy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import sha256_file, stable_json_hash
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_v31_ablation import evaluate_config_on_subject_features

DEVELOPMENT_SUBJECTS = ("gB-9", "gZ-36", "gC-14")
CONSUMED_HOLDOUTS = ("gZ-37", "gE-28")
REQUIRED_RECORD_FIELDS = {
    "timestamp_seconds",
    "frame_index",
    "body_phone_conf",
    "face_phone_conf",
    "hands_phone_conf",
    "body_available",
    "face_available",
    "hands_available",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def discover_external_dmd_root() -> Path:
    """Support the tracked plural path and an explicitly restored singular data directory."""
    candidates = (Path("datasets/external_dmd"), Path("dataset/external_dmd"))
    for candidate in candidates:
        if (candidate / "derivatives").is_dir() and (candidate / "subjects").is_dir():
            return candidate
    return candidates[0]


def _annotation_path(subjects_dir: Path, subject_id: str) -> Path | None:
    matches = sorted((subjects_dir / subject_id).glob("*_ann_distraction.json"))
    return matches[0] if len(matches) == 1 else None


def preflight(
    derivatives_dir: Path,
    config_path: Path,
    subjects_dir: Path | None = None,
) -> dict[str, Any]:
    subjects_dir = subjects_dir or derivatives_dir.parent / "subjects"
    blockers: list[str] = []
    inputs: list[dict[str, Any]] = []
    for subject_id in DEVELOPMENT_SUBJECTS:
        assert_not_consumed_holdout_for_development(subject_id, caller_action="v4_preflight")
        path = derivatives_dir / f"v3_features_{subject_id}.json"
        annotation = _annotation_path(subjects_dir, subject_id)
        if not path.is_file():
            blockers.append(f"missing cached development features: {path}")
            continue
        if annotation is None:
            blockers.append(
                f"expected exactly one development annotation for {subject_id}: "
                f"{subjects_dir / subject_id}"
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"unreadable cached development features {path}: {exc}")
            continue
        records = payload.get("records", [])
        if not isinstance(records, list) or not records:
            blockers.append(f"cached development features contain no records: {path}")
            continue
        malformed_rows = [
            index
            for index, record in enumerate(records)
            if not isinstance(record, dict) or REQUIRED_RECORD_FIELDS - set(record)
        ]
        if malformed_rows:
            blockers.append(
                f"{path} has {len(malformed_rows)} rows missing required fields; "
                f"first indexes: {malformed_rows[:5]}"
            )
        inputs.append(
            {
                "subject_id": subject_id,
                "features_path": str(path),
                "features_sha256": sha256_file(path),
                "annotation_path": str(annotation),
                "annotation_sha256": sha256_file(annotation),
                "records": len(records),
                "duration_seconds": payload.get("duration_seconds"),
            }
        )

    if not config_path.is_file():
        blockers.append(f"missing V4 config: {config_path}")
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "dataset_role": "DEVELOPMENT_ONLY",
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "consumed_holdouts_forbidden": list(CONSUMED_HOLDOUTS),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "inputs": inputs,
        "input_set_sha256": stable_json_hash(inputs),
        "blockers": blockers,
        "generalization_claim": "NOT_ESTABLISHED",
    }


def _experiment_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    temporal = dict(config.get("temporal", {}))
    v4_temporal = {
        key: temporal[key]
        for key in (
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
            "strict_phone_context",
        )
        if key in temporal
    }
    baseline = {
        "auxiliary_rescue_mode": False,
        "occlusion_bridge_seconds": 4.0,
        "temporal_window": 30,
        "merge_gap_seconds": 3.0,
        "enable_face": True,
        "enable_hands": True,
        "enable_pose": True,
        "strict_phone_context": False,
    }
    return {
        "V3_BASELINE": {"name": "Frozen V3 development baseline", **baseline},
        "V4_GATED": {
            "name": "BODY-primary gated auxiliary fusion",
            **baseline,
            "auxiliary_rescue_mode": True,
            "enable_pose": False,
            "occlusion_bridge_seconds": 2.0,
        },
        "V4_POSE_GATED": {
            "name": "Pose-conditioned auxiliary rescue",
            **baseline,
            "auxiliary_rescue_mode": True,
            "occlusion_bridge_seconds": 2.0,
        },
        "V4_TEMPORAL": {
            "name": "V4 temporal policy with strict context",
            **baseline,
            **v4_temporal,
            "temporal_window": 20,
            "merge_gap_seconds": 5.0,
        },
        "V4_COMBINED": {
            "name": "Gated fusion, pose rescue, strict context, and V4 temporal policy",
            **baseline,
            **v4_temporal,
            "auxiliary_rescue_mode": True,
            "temporal_window": 20,
            "merge_gap_seconds": 5.0,
        },
    }


def _pool_results(per_subject: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["true_positives"] for item in per_subject.values())
    fp = sum(item["false_positives"] for item in per_subject.values())
    fn = sum(item["false_negatives"] for item in per_subject.values())
    total_gt = sum(item["total_gt_events"] for item in per_subject.values())
    minutes = sum(item["evaluated_minutes"] for item in per_subject.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision * 100.0, 1),
        "recall": round(recall * 100.0, 1),
        "f1": round(f1 * 100.0, 1),
        "false_alarms_per_minute": round(fp / minutes, 3) if minutes else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_ground_truth_events": total_gt,
        "evaluated_minutes": round(minutes, 2),
    }


def run_experiments(
    derivatives_dir: Path,
    subjects_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    check = preflight(derivatives_dir, config_path, subjects_dir)
    if check["status"] != "READY":
        return check

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    results: list[dict[str, Any]] = []
    for experiment_id, spec in _experiment_specs(config).items():
        per_subject: dict[str, dict[str, Any]] = {}
        for subject_id in DEVELOPMENT_SUBJECTS:
            assert_not_consumed_holdout_for_development(subject_id, caller_action="v4_ablation")
            feature_path = derivatives_dir / f"v3_features_{subject_id}.json"
            cached = json.loads(feature_path.read_text(encoding="utf-8"))
            per_subject[subject_id] = evaluate_config_on_subject_features(
                subject_id,
                subjects_dir / subject_id,
                cached,
                spec,
                include_evidence=True,
            )
        results.append(
            {
                "experiment_id": experiment_id,
                "name": spec["name"],
                "spec": spec,
                "pooled_metrics": _pool_results(per_subject),
                "per_subject": per_subject,
            }
        )

    eligible = [
        item
        for item in results
        if item["experiment_id"] != "V3_BASELINE"
        and item["pooled_metrics"]["false_alarms_per_minute"] <= 0.30
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["pooled_metrics"]["f1"],
            item["pooled_metrics"]["recall"],
            -item["pooled_metrics"]["false_alarms_per_minute"],
        ),
        default=None,
    )
    return {
        **check,
        "status": "COMPLETED",
        "matching_policy": "training.dmd.temporal_eval.TemporalMatchingPolicy defaults",
        "selection_rule": (
            "Highest development F1 among V4 candidates with FA/min <= 0.30; then recall; "
            "then lower FA/min"
        ),
        "experiments": results,
        "selected_candidate": selected["experiment_id"] if selected else None,
        "selection_status": "SELECTED" if selected else "NO_CANDIDATE_MET_FALSE_ALARM_GATE",
        "skipped_ablations": {
            "V4_RELIABILITY_WEIGHTED": (
                "Cached features have view availability but no per-frame visibility, occlusion, "
                "or camera-calibration reliability labels; weighting would be ungrounded."
            )
        },
        "new_independent_holdout": "NOT_AVAILABLE",
        "generalization_claim": "NOT_ESTABLISHED",
    }


def main() -> None:
    data_root = discover_external_dmd_root()
    parser = argparse.ArgumentParser(description="Run governed Roadwatch V4 development ablations")
    parser.add_argument("--derivatives-dir", type=Path, default=data_root / "derivatives")
    parser.add_argument("--subjects-dir", type=Path, default=data_root / "subjects")
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v4_candidate.yaml")
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("reports/v4_upgrade/experiment_preflight.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v4_upgrade/v4_development_results.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    check = preflight(args.derivatives_dir, args.config, args.subjects_dir)
    args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_output.write_text(json.dumps(check, indent=2), encoding="utf-8")
    if args.preflight_only or check["status"] != "READY":
        result = check
    else:
        result = run_experiments(args.derivatives_dir, args.subjects_dir, args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
