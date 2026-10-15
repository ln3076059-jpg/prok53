"""Stage-level false-negative diagnostics for the frozen V4 development result."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.fusion import MultiViewFusionEngine
from training.dmd.pose_features import PoseFeatureVector
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.temporal_eval import TemporalMatchingPolicy, match_temporal_events


def _overlaps(record_time: float, event: dict[str, Any]) -> bool:
    return float(event["start_seconds"]) <= record_time <= float(event["end_seconds"])


def _event_overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return min(float(a["end_seconds"]), float(b["end_seconds"])) > max(
        float(a["start_seconds"]), float(b["start_seconds"])
    )


def diagnose_v4_false_negatives(
    derivatives_dir: Path,
    subjects_dir: Path,
    v4_results_path: Path,
) -> dict[str, Any]:
    v4 = json.loads(v4_results_path.read_text(encoding="utf-8"))
    selected_id = v4["selected_candidate"]
    selected = next(item for item in v4["experiments"] if item["experiment_id"] == selected_id)
    spec = selected["spec"]
    cases: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    stage_reached: Counter[str] = Counter()

    for subject_id in DEVELOPMENT_SUBJECTS:
        cached = json.loads(
            (derivatives_dir / f"v3_features_{subject_id}.json").read_text(encoding="utf-8")
        )
        annotation_path = next((subjects_dir / subject_id).glob("*_ann_distraction.json"))
        annotation = parse_dmd_openlabel(annotation_path)
        truths = [item.to_dict() for item in annotation.phone_intervals]
        evaluation = selected["per_subject"][subject_id]["evaluation_evidence"]
        predictions = evaluation["predicted_events"]
        matches, _, _ = match_temporal_events(truths, predictions, TemporalMatchingPolicy())
        matched_keys = {
            (
                float(item["ground_truth"]["start_seconds"]),
                float(item["ground_truth"]["end_seconds"]),
            )
            for item in matches
        }

        fusion = MultiViewFusionEngine(
            phone_conf_threshold=0.25,
            temporal_window=int(spec["temporal_window"]),
            enable_face_fusion=bool(spec["enable_face"]),
            enable_hands_fusion=bool(spec["enable_hands"]),
            enable_pose_fusion=bool(spec["enable_pose"]),
            auxiliary_rescue_mode=bool(spec["auxiliary_rescue_mode"]),
        )
        enriched: list[dict[str, Any]] = []
        for record in cached["records"]:
            pose = PoseFeatureVector(*record["pose_features"]) if record.get("pose_features") else None
            fused = fusion.fuse_frame(
                timestamp_seconds=record["timestamp_seconds"],
                frame_index=record["frame_index"],
                body_phone_conf=record["body_phone_conf"],
                face_phone_conf=record["face_phone_conf"],
                hands_phone_conf=record["hands_phone_conf"],
                body_pose_features=pose,
                body_available=record["body_available"],
                face_available=record["face_available"],
                hands_available=record["hands_available"],
            )
            enriched.append({**record, **fused.to_dict()})

        for event_index, truth in enumerate(truths, start=1):
            records = [item for item in enriched if _overlaps(item["timestamp_seconds"], truth)]
            max_body = max((float(item["body_phone_conf"]) for item in records), default=0.0)
            max_face = max((float(item["face_phone_conf"]) for item in records), default=0.0)
            max_hands = max((float(item["hands_phone_conf"]) for item in records), default=0.0)
            max_any = max(max_body, max_face, max_hands)
            max_fused = max((float(item["fused_score"]) for item in records), default=0.0)
            max_behavior = max(
                (max(float(item["pose_call_score"]), float(item["pose_text_score"])) for item in records),
                default=0.0,
            )
            sustained_candidate = any(
                float(a["fused_score"]) >= 0.18 and float(b["fused_score"]) >= 0.18
                for a, b in zip(records, records[1:], strict=False)
            )
            overlapping_predictions = [item for item in predictions if _event_overlaps(item, truth)]
            key = (float(truth["start_seconds"]), float(truth["end_seconds"]))
            matched = key in matched_keys

            checks = {
                "phone_object_detector": max_any > 0.0,
                "phone_candidate_threshold": max_any >= 0.10,
                "person_driver_association": None,
                "body_view": max_body >= 0.10,
                "face_view": max_face >= 0.10,
                "hands_view": max_hands >= 0.10,
                "cross_view_fusion": max_fused >= 0.18,
                "behavioral_classification": max_behavior >= 0.30,
                "temporal_candidate": sustained_candidate,
                "event_activation": bool(overlapping_predictions),
                "event_matching": matched,
            }
            for stage, passed in checks.items():
                if passed is True:
                    stage_reached[stage] += 1

            if not checks["phone_object_detector"]:
                failure_stage = "PHONE_OBJECT_DETECTOR"
            elif not checks["phone_candidate_threshold"]:
                failure_stage = "PHONE_CANDIDATE_THRESHOLD"
            elif not checks["cross_view_fusion"]:
                failure_stage = "CROSS_VIEW_FUSION"
            elif not checks["behavioral_classification"]:
                failure_stage = "BEHAVIORAL_CLASSIFICATION"
            elif not checks["temporal_candidate"]:
                failure_stage = "TEMPORAL_CANDIDATE"
            elif not checks["event_activation"]:
                failure_stage = "EVENT_ACTIVATION"
            elif not matched:
                failure_stage = "EVENT_MATCHING"
            else:
                failure_stage = "MATCHED_TRUE_POSITIVE"
            stage_counts[failure_stage] += 1
            cases.append(
                {
                    "case_id": f"{subject_id}-phone-{event_index:02d}",
                    "subject_id": subject_id,
                    "truth": truth,
                    "v4_result": "TP" if matched else "FN",
                    "primary_failure_stage": failure_stage,
                    "checks": checks,
                    "evidence": {
                        "sampled_records": len(records),
                        "max_body_phone_conf": round(max_body, 4),
                        "max_face_phone_conf": round(max_face, 4),
                        "max_hands_phone_conf": round(max_hands, 4),
                        "max_any_phone_conf": round(max_any, 4),
                        "max_fused_score": round(max_fused, 4),
                        "max_behavior_score": round(max_behavior, 4),
                        "overlapping_activated_events": overlapping_predictions,
                    },
                    "observability_limits": [
                        "Person/driver association is not serialized in the V3 feature cache.",
                        "Availability flags do not prove visibility or absence of occlusion.",
                    ],
                }
            )

    total = len(cases)
    false_negatives = [item for item in cases if item["v4_result"] == "FN"]
    stages = [
        "phone_object_detector",
        "phone_candidate_threshold",
        "person_driver_association",
        "body_view",
        "face_view",
        "hands_view",
        "cross_view_fusion",
        "behavioral_classification",
        "temporal_candidate",
        "event_activation",
        "event_matching",
    ]
    return {
        "status": "MEASURED_ON_ROADWATCH",
        "source_artifact": str(v4_results_path),
        "selected_v4_candidate": selected_id,
        "events": total,
        "true_positives": total - len(false_negatives),
        "false_negatives": len(false_negatives),
        "primary_failure_stage_counts_all_events": dict(stage_counts),
        "stage_level_event_recall_upper_bounds": {
            stage: {
                "events_reached": stage_reached.get(stage, 0),
                "total_events": total,
                "recall": round(stage_reached.get(stage, 0) / total, 4) if total else 0.0,
                "status": "NOT_OBSERVABLE_IN_CACHE" if stage == "person_driver_association" else "MEASURED_ON_ROADWATCH",
            }
            for stage in stages
        },
        "false_negative_cases": false_negatives,
        "all_cases": cases,
        "conclusion": (
            "The cache can quantify detector/fusion/temporal evidence, but cannot establish "
            "association or visual occlusion causes. Events with no cached object evidence require "
            "new RGB feature extraction or an independent video action model."
        ),
    }
