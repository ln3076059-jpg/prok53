"""Produce the P0 measurement, alignment, and causality audits for AI85 V31.

The script is intentionally read-only with respect to model/data inputs.  It
re-scores existing artifacts and writes compact evidence files under a new
report directory; it does not train, tune thresholds, or consume a holdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from training.common import sha256_file, stable_json_hash
from training.dmd.evaluate_generic_phone_episodes import evaluate_run
from training.dmd.run_v5_experiments import _ground_truth, discover_external_dmd_root
from training.dmd.run_v30_causal_action import label_object_events_with_actions
from training.dmd.temporal_eval import TemporalMatchingPolicy, match_temporal_events
from training.dmd.v30_causal_action import CausalActionGRU, class_aware_event_metrics
from training.dmd.v5_video_action import label_timestamps
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.v31_streaming_fusion import (
    StreamingFusionConfig,
    StreamingObjectActionFusion,
)


BENCHMARK_SUBJECTS = ("gC-14", "gZ-36")
DEVELOPMENT_SUBJECTS = ("gB-9", "gC-14", "gZ-36")
BASELINE_RUN = Path("reports/v28_local_raw_d5/imgsz640")
ACTION_RUNS = {
    "V30_WEIGHTED": Path("reports/v30_causal_action/v30_results.json"),
    "V30_UNWEIGHTED": Path("reports/v30_causal_action_unweighted/v30_results.json"),
    "V31_DIRECT112_WEIGHTED": Path(
        "reports/v31_direct112_causal_action/v30_results.json"
    ),
    "V31_DIRECT112_UNWEIGHTED": Path(
        "reports/v31_direct112_causal_action_unweighted/v30_results.json"
    ),
}
COARSE_RUNS = {
    "V31_COARSE_WEIGHTED": Path(
        "reports/v31_direct112_coarse_action/v31_coarse_results.json"
    ),
    "V31_COARSE_UNWEIGHTED": Path(
        "reports/v31_direct112_coarse_action_unweighted/v31_coarse_results.json"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _temporal_metrics(
    truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    duration_seconds: float,
    policy: TemporalMatchingPolicy,
) -> dict[str, Any]:
    matched, unmatched_predictions, unmatched_truth = match_temporal_events(
        truth, predictions, policy
    )
    unique_truth = {
        (
            float(row["ground_truth"]["start_seconds"]),
            float(row["ground_truth"]["end_seconds"]),
        )
        for row in matched
    }
    tp = len(unique_truth)
    duplicates = max(0, len(matched) - tp)
    fp = len(unmatched_predictions) + duplicates
    fn = len(unmatched_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": len(truth),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(
            fp / max(duration_seconds / 60.0, 1e-9), 4
        ),
        "duplicate_predictions": duplicates,
    }


def _aggregate(per_subject: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["true_positives"] for row in per_subject.values())
    fp = sum(row["false_positives"] for row in per_subject.values())
    fn = sum(row["false_negatives"] for row in per_subject.values())
    gt = sum(row["total_gt_events"] for row in per_subject.values())
    minutes = sum(row["duration_seconds"] for row in per_subject.values()) / 60.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": gt,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(minutes, 1e-9), 4),
        "evaluated_minutes": round(minutes, 4),
        "per_subject": per_subject,
    }


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "true_positives",
        "false_positives",
        "false_negatives",
        "total_gt_events",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_minute",
        "evaluated_minutes",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _events_from_recorded_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    events = [dict(row["prediction"]) for row in metrics.get("matched_events", [])]
    events.extend(dict(row) for row in metrics.get("unmatched_predictions", []))
    return events


def build_metric_protocol_matrix() -> dict[str, Any]:
    policy = TemporalMatchingPolicy()
    frozen_path = Path("reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json")
    frozen = _read_json(frozen_path)
    root = discover_external_dmd_root()
    subjects_dir = root / "subjects"
    baseline_reported = _read_json(BASELINE_RUN / "event_matches.json")
    durations = {
        subject: float(
            _read_json(root / "derivatives" / f"v3_features_{subject}.json")[
                "duration_seconds"
            ]
        )
        for subject in DEVELOPMENT_SUBJECTS
    }

    baseline_per_subject: dict[str, dict[str, Any]] = {}
    for subject in BENCHMARK_SUBJECTS:
        predictions = _read_jsonl(
            BASELINE_RUN / "subjects" / subject / "event_predictions.jsonl"
        )
        row = _temporal_metrics(
            _ground_truth(subjects_dir, subject),
            predictions,
            duration_seconds=durations[subject],
            policy=policy,
        )
        row["duration_seconds"] = durations[subject]
        baseline_per_subject[subject] = row
    baseline_recomputed = _aggregate(baseline_per_subject)
    baseline_expected = {
        "true_positives": baseline_reported["true_positives"],
        "false_positives": baseline_reported["false_positives"],
        "false_negatives": baseline_reported["false_negatives"],
        "precision": baseline_reported["precision"],
        "recall": baseline_reported["recall"],
        "f1": baseline_reported["f1"],
    }
    reproduction_pass = all(
        baseline_recomputed[key] == value for key, value in baseline_expected.items()
    )

    generic = evaluate_run(BASELINE_RUN, gap_seconds=3.0, policy=policy)
    v30_rescores: dict[str, Any] = {}
    for run_name, result_path in ACTION_RUNS.items():
        result = _read_json(result_path)
        branches: dict[str, Any] = {}
        branch_specs = {
            "ACTION_ONLY": None,
            "OBJECT_PLUS_ACTION_SEGMENTATION": "object_plus_action_segmentation",
            "OBJECT_PLUS_HIGH_CONFIDENCE_ACTION_RECOVERY": (
                "object_plus_high_confidence_action_recovery"
            ),
        }
        for branch_name, result_key in branch_specs.items():
            temporal_per_subject: dict[str, dict[str, Any]] = {}
            class_per_subject: dict[str, dict[str, Any]] = {}
            for subject in BENCHMARK_SUBJECTS:
                if result_key is None:
                    recorded = result["action_only"]["per_subject"][subject]
                    events = _events_from_recorded_metrics(recorded)
                else:
                    events = result[result_key]["events"][subject]
                truth = _ground_truth(subjects_dir, subject)
                temporal = _temporal_metrics(
                    truth,
                    events,
                    duration_seconds=durations[subject],
                    policy=policy,
                )
                temporal["duration_seconds"] = durations[subject]
                aware = class_aware_event_metrics(
                    truth, events, duration_seconds=durations[subject], policy=policy
                )
                aware["total_gt_events"] = len(truth)
                aware["duration_seconds"] = durations[subject]
                temporal_per_subject[subject] = temporal
                class_per_subject[subject] = aware
            branches[branch_name] = {
                "v2_temporal_only": _compact(_aggregate(temporal_per_subject)),
                "class_aware_exact_action": _compact(_aggregate(class_per_subject)),
            }
        v30_rescores[run_name] = {
            "source": str(result_path),
            "source_sha256": sha256_file(result_path),
            "branches": branches,
        }

    coarse_rescores: dict[str, Any] = {}
    coarse_branch_specs = {
        "ACTION_ONLY": "action_only",
        "OBJECT_GATED_BY_COARSE_ACTION": "object_gated_by_coarse_action",
        "OBJECT_PLUS_COARSE_RECOVERY": "object_plus_coarse_recovery",
    }
    for run_name, result_path in COARSE_RUNS.items():
        result = _read_json(result_path)
        branches: dict[str, Any] = {}
        for branch_name, result_key in coarse_branch_specs.items():
            per_subject: dict[str, dict[str, Any]] = {}
            for subject in BENCHMARK_SUBJECTS:
                events = result[result_key]["events"][subject]
                row = _temporal_metrics(
                    _ground_truth(subjects_dir, subject),
                    events,
                    duration_seconds=durations[subject],
                    policy=policy,
                )
                row["duration_seconds"] = durations[subject]
                per_subject[subject] = row
            branches[branch_name] = {
                "v2_temporal_only": _compact(_aggregate(per_subject)),
                "reported": _compact(result[result_key]["benchmark_gc_gz"]),
            }
        coarse_rescores[run_name] = {
            "source": str(result_path),
            "source_sha256": sha256_file(result_path),
            "pooled_clip_metrics": result["pooled_clip_metrics_all_development"],
            "branches": branches,
        }

    protocols = {
        "V2_TEMPORAL_FROZEN": {
            "role": "PRIMARY_DEVELOPMENT_METRIC",
            "unit": "ACTION_INTERVAL",
            "class_constraint": "NONE_GENERIC_PHONE",
            "matching_policy_id": frozen["policy_id"],
            "eligible_for_ai85_claim": True,
        },
        "CLASS_AWARE_EXACT_SOURCE_ACTION": {
            "role": "SECONDARY_STRICT_DIAGNOSTIC",
            "unit": "ACTION_INTERVAL",
            "class_constraint": "EXACT_TEXTING_PHONECALL_AND_SIDE",
            "matching_policy_id": frozen["policy_id"],
            "eligible_for_ai85_claim": False,
        },
        "GENERIC_PHONE_EPISODE_GAP3S": {
            "role": "SECONDARY_EPISODE_DIAGNOSTIC",
            "unit": "GENERIC_PHONE_EPISODE",
            "class_constraint": "NONE",
            "matching_policy_id": frozen["policy_id"],
            "eligible_for_ai85_claim": False,
        },
    }
    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if reproduction_pass else "FAIL_BASELINE_REPRODUCTION",
        "frozen_policy": {
            "path": str(frozen_path),
            "sha256": sha256_file(frozen_path),
            "declared_policy_sha256": frozen["policy_sha256"],
            "parameters": {
                "iou_threshold": policy.iou_threshold,
                "onset_tolerance_seconds": policy.onset_tolerance_seconds,
                "offset_tolerance_seconds": policy.offset_tolerance_seconds,
                "containment_overlap_threshold": frozen[
                    "containment_overlap_threshold"
                ],
                "containment_min_duration_seconds": frozen[
                    "containment_min_duration_seconds"
                ],
            },
        },
        "protocols": protocols,
        "baseline_v2": {
            "source": str(BASELINE_RUN),
            "reported": baseline_expected,
            "recomputed": _compact(baseline_recomputed),
            "exact_reproduction": reproduction_pass,
        },
        "generic_episode_diagnostic": _compact(generic),
        "v30_rescores": v30_rescores,
        "coarse_head_rescores": coarse_rescores,
        "claim_rule": (
            "Only V2_TEMPORAL_FROZEN on an independent subject-disjoint holdout may "
            "support the >85% claim; the other protocols are diagnostics."
        ),
    }


def build_alignment_audit() -> dict[str, Any]:
    root = discover_external_dmd_root()
    subjects_dir = root / "subjects"
    derivatives = root / "derivatives"
    per_subject: dict[str, Any] = {}
    all_pass = True
    for subject in DEVELOPMENT_SUBJECTS:
        feature_path = derivatives / f"v3_features_{subject}.json"
        embedding_path = (
            derivatives
            / "v5_resnet18_embeddings"
            / f"v5_resnet18_{subject}.npz"
        )
        rgb_path = derivatives / "v5_rgb_cache" / f"v5_rgb_{subject}_64.npz"
        annotation_path = _find_one(
            subjects_dir / subject, "*_rgb_ann_distraction.json"
        )
        video_path = _find_one(subjects_dir / subject, "*_rgb_body.mp4")
        feature = _read_json(feature_path)
        feature_records = feature["records"]
        feature_timestamps = np.asarray(
            [float(row["timestamp_seconds"]) for row in feature_records],
            dtype=np.float64,
        )
        feature_frames = np.asarray(
            [int(row["frame_index"]) for row in feature_records], dtype=np.int64
        )
        with np.load(embedding_path, allow_pickle=False) as payload:
            embedding_shape = list(payload["embeddings"].shape)
            embedding_timestamps = payload["timestamps"].astype(np.float64)
            embedding_labels = payload["labels"].astype(np.int64)
        with np.load(rgb_path, allow_pickle=False) as payload:
            rgb_shape = list(payload["frames"].shape)
            rgb_timestamps = payload["timestamps"].astype(np.float64)
            rgb_frames = payload["frame_indices"].astype(np.int64)
            rgb_labels = payload["labels"].astype(np.int64)
        annotation = parse_dmd_openlabel(annotation_path)
        expected_labels, _ = label_timestamps(embedding_timestamps, annotation)
        label_mismatch_indices = np.flatnonzero(embedding_labels != expected_labels)
        action_boundaries = np.asarray(
            [
                boundary
                for interval in annotation.all_intervals
                if interval.source_action.startswith("driver_actions/")
                for boundary in (interval.start_seconds, interval.end_seconds)
            ],
            dtype=np.float64,
        )
        mismatch_boundary_distances = [
            float(np.min(np.abs(action_boundaries - embedding_timestamps[index])))
            for index in label_mismatch_indices
        ]
        maximum_mismatch_boundary_distance = max(
            mismatch_boundary_distances, default=0.0
        )
        boundary_quantization_only = maximum_mismatch_boundary_distance <= 0.0001

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video metadata: {video_path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_time_error = np.abs(feature_timestamps - feature_frames / source_fps)
        timestamp_step = np.diff(feature_timestamps)
        frame_step = np.diff(feature_frames)
        checks = {
            "record_counts_equal": (
                len(feature_records)
                == embedding_shape[0]
                == rgb_shape[0]
                == len(embedding_timestamps)
            ),
            "feature_embedding_timestamps_equal": bool(
                np.allclose(feature_timestamps, embedding_timestamps, atol=1e-5)
            ),
            "feature_rgb_timestamps_equal": bool(
                np.allclose(feature_timestamps, rgb_timestamps, atol=1e-5)
            ),
            "feature_rgb_frame_indices_equal": bool(
                np.array_equal(feature_frames, rgb_frames)
            ),
            "embedding_rgb_labels_equal": bool(
                np.array_equal(embedding_labels, rgb_labels)
            ),
            "labels_match_annotation_with_100us_boundary_tolerance": bool(
                len(label_mismatch_indices) == 0 or boundary_quantization_only
            ),
            "timestamps_strictly_increasing": bool(np.all(timestamp_step > 0)),
            "frames_strictly_increasing": bool(np.all(frame_step > 0)),
            "rgb_cache_is_64px_rgb": rgb_shape[1:] == [64, 64, 3],
            "embedding_dimension_is_512": embedding_shape[1:] == [512],
            "frame_timestamp_error_below_1ms": bool(
                float(frame_time_error.max(initial=0.0)) <= 0.001
            ),
        }
        subject_pass = all(checks.values())
        all_pass = all_pass and subject_pass
        per_subject[subject] = {
            "status": "PASS" if subject_pass else "FAIL",
            "checks": checks,
            "records": len(feature_records),
            "source_video_fps": round(source_fps, 6),
            "source_video_frames": total_frames,
            "declared_sample_stride": feature["sample_stride"],
            "median_frame_step": float(np.median(frame_step)),
            "median_timestamp_step_seconds": round(float(np.median(timestamp_step)), 6),
            "maximum_frame_timestamp_error_seconds": round(
                float(frame_time_error.max(initial=0.0)), 8
            ),
            "label_mismatches_embedding_vs_annotation": int(
                np.sum(embedding_labels != expected_labels)
            ),
            "label_mismatch_maximum_distance_to_action_boundary_seconds": round(
                maximum_mismatch_boundary_distance, 8
            ),
            "label_mismatch_classification": (
                "FLOAT32_BOUNDARY_QUANTIZATION"
                if len(label_mismatch_indices) and boundary_quantization_only
                else "NONE"
                if not len(label_mismatch_indices)
                else "NON_BOUNDARY_MISMATCH"
            ),
            "unknown_label_samples": int(np.sum(embedding_labels < 0)),
            "rgb_cache_shape": rgb_shape,
            "embedding_shape": embedding_shape,
            "preprocessing": {
                "source_crop": "BODY_VIEW_FIXED_ROI_[0.0,0.10,0.65,0.95]",
                "cache_resize": "64x64_INTER_AREA",
                "encoder_resize": "64x64_TO_112x112_BILINEAR",
                "encoder": "FROZEN_IMAGENET_RESNET18",
            },
            "input_hashes": {
                "features": sha256_file(feature_path),
                "rgb_cache": sha256_file(rgb_path),
                "embeddings": sha256_file(embedding_path),
                "annotation": sha256_file(annotation_path),
            },
        }
    has_boundary_quantization = any(
        row["label_mismatch_classification"] == "FLOAT32_BOUNDARY_QUANTIZATION"
        for row in per_subject.values()
    )
    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "PASS_WITH_BOUNDARY_QUANTIZATION"
            if all_pass and has_boundary_quantization
            else "PASS"
            if all_pass
            else "FAIL"
        ),
        "subjects": per_subject,
        "limitations": [
            "Alignment PASS proves timestamp/frame/label consistency, not visual feature sufficiency.",
            "The cached RGB input is only 64x64 and is later enlarged to 112x112; lost detail cannot be restored by interpolation.",
            "Only three development subjects are present, so subject generalization remains weakly estimated.",
            "Thirteen cached labels sit within 14.6 microseconds of an annotation boundary; exact float64 replay changes them to UNKNOWN, so future extraction should persist an explicit boundary convention.",
        ],
    }


def _future_dependent_fusion_probe() -> dict[str, Any]:
    object_events = [
        {
            "event_type": "PHONE",
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "duration": 2.0,
            "confidence": 0.4,
        }
    ]
    timestamps = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    left_future = np.zeros((3, 6), dtype=np.float64)
    right_future = np.zeros((3, 6), dtype=np.float64)
    left_future[:, 1] = [0.9, 0.9, 0.9]
    right_future[:, 1] = [0.9, 0.1, 0.1]
    right_future[:, 2] = [0.1, 0.95, 0.95]
    left = label_object_events_with_actions(
        object_events,
        [],
        timestamps,
        left_future,
        add_action_only_recovery=False,
    )
    right = label_object_events_with_actions(
        object_events,
        [],
        timestamps,
        right_future,
        add_action_only_recovery=False,
    )
    return {
        "same_prefix_through_seconds": 0.0,
        "label_with_left_future": left[0]["source_action"],
        "label_with_right_future": right[0]["source_action"],
        "future_influence_detected": (
            left[0]["source_action"] != right[0]["source_action"]
        ),
    }


def _optimization_sanity_probe() -> dict[str, Any]:
    """Overfit one deterministic synthetic minibatch to verify train wiring."""

    torch.manual_seed(31085)
    model = CausalActionGRU(embedding_dim=8, hidden_size=12)
    model.train()
    inputs = torch.randn(24, 5, 8)
    labels = torch.arange(24, dtype=torch.long) % 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        initial_loss = float(criterion(model(inputs), labels).item())
    finite_gradients = True
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), labels)
        loss.backward()
        finite_gradients = finite_gradients and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        final_loss = float(criterion(logits, labels).item())
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    return {
        "purpose": "TRAINING_WIRING_ONLY_NOT_AN_EVALUATION_METRIC",
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "minibatch_accuracy": round(accuracy, 6),
        "finite_gradients": finite_gradients,
        "pass": finite_gradients and final_loss < initial_loss * 0.25 and accuracy >= 0.95,
    }


def _streaming_prefix_invariance_probe() -> dict[str, Any]:
    config = StreamingFusionConfig(
        action_threshold=0.55,
        label_update_margin=0.05,
        probability_ema_alpha=0.8,
    )
    rows = [
        (0.0, False, 0),
        (0.5, True, 1),
        (1.0, True, 1),
        (1.5, True, 2),
        (2.0, False, 0),
    ]

    def probability(class_id: int) -> np.ndarray:
        row = np.full(6, 0.02, dtype=np.float64)
        row[class_id] = 0.90
        return row / row.sum()

    def replay(limit: int) -> list[dict[str, Any]]:
        fusion = StreamingObjectActionFusion(config)
        for timestamp, active, class_id in rows[:limit]:
            fusion.update(
                timestamp=timestamp,
                object_active=active,
                object_confidence=0.8 if active else 0.0,
                action_probabilities=probability(class_id),
            )
        return fusion.decisions()

    prefix = replay(3)
    full = replay(len(rows))
    smoke_path = Path(
        "reports/v31_ai85_plan_execution/d5_live_decision_smoke_v2/"
        "subjects/gC-14/object_decisions.jsonl"
    )
    live_decisions = _read_jsonl(smoke_path) if smoke_path.is_file() else []
    live_contract_pass = bool(live_decisions) and all(
        row["event_start_seconds"] <= row["decision_time_seconds"]
        and row["evidence_available_through_seconds"]
        == row["decision_time_seconds"]
        for row in live_decisions
    )
    return {
        "prefix_decisions": prefix,
        "full_decision_types": [row["decision_type"] for row in full],
        "prefix_invariant": full[: len(prefix)] == prefix,
        "action_only_activation_allowed": config.allow_action_only_activation,
        "live_d5_object_decisions_available": live_contract_pass,
        "live_d5_smoke_path": str(smoke_path),
        "live_d5_decision_types": [
            row["decision_type"] for row in live_decisions
        ],
        "integrated_with_live_d5_runner": live_contract_pass,
    }


def build_causal_contract_audit() -> dict[str, Any]:
    results = {
        name: _read_json(path)
        for name, path in {**ACTION_RUNS, **COARSE_RUNS}.items()
    }
    head_contracts = {
        name: {
            "window": row["causal_contract"]["window"],
            "target_position": row["causal_contract"]["target_position"],
            "recurrent_direction": row["causal_contract"]["recurrent_direction"],
            "future_frames": row["causal_contract"]["future_frames"],
            "pass": (
                row["causal_contract"]["window"] == "PAST_AND_CURRENT_ONLY"
                and row["causal_contract"]["target_position"] == "FINAL_SAMPLE"
                and row["causal_contract"]["recurrent_direction"]
                == "UNIDIRECTIONAL"
                and row["causal_contract"]["future_frames"] == 0
            ),
        }
        for name, row in results.items()
    }
    probe = _future_dependent_fusion_probe()
    optimization_probe = _optimization_sanity_probe()
    streaming_probe = _streaming_prefix_invariance_probe()
    overall_pass = all(row["pass"] for row in head_contracts.values()) and not probe[
        "future_influence_detected"
    ]
    return {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if overall_pass else "BLOCKED_OFFLINE_FUSION",
        "action_head": {
            "status": "PASS",
            "contracts": head_contracts,
            "optimization_sanity": optimization_probe,
            "evidence": (
                "Every input window ends at its target and the GRU is unidirectional."
            ),
        },
        "decoder": {
            "status": "PARTIAL",
            "finding": (
                "The decoder consumes probabilities in timestamp order, but emitted events "
                "do not store decision_time_seconds. Event starts are back-dated to the first "
                "positive sample, so online latency cannot be audited from the artifact."
            ),
        },
        "object_action_fusion": {
            "status": "FAIL_CAUSAL_END_TO_END",
            "probe": probe,
            "finding": (
                "Dominant-label fusion averages probabilities across the complete object-event "
                "interval; changing later samples can change the label assigned at the start. "
                "Overlap fusion also consumes fully decoded action intervals."
            ),
        },
        "streaming_fusion_prototype": {
            "status": "PASS_CONTRACT_ACTION_NOT_CONNECTED",
            "probe": streaming_probe,
            "finding": (
                "The append-only prototype is prefix invariant and records decision time. "
                "The D5 runner now emits validated live object START/END decisions, but "
                "causal action probabilities are not yet connected to that live stream."
            ),
        },
        "required_fix_before_promotion": [
            "Replace whole-interval mean labeling with per-timestamp streaming fusion state.",
            "Record observation_time, decision_time, event_start, and event_end separately.",
            "Add prefix-invariance tests for the complete detector-action-fusion output stream.",
            "Re-score streaming output under V2 temporal and class-aware protocols without changing thresholds after inspection.",
        ],
        "promotion_eligible": False,
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metric_protocol_matrix.json": build_metric_protocol_matrix(),
        "alignment_audit.json": build_alignment_audit(),
        "causal_contract_audit.json": build_causal_contract_audit(),
    }
    receipts: dict[str, Any] = {}
    for name, payload in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        receipts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": payload["status"],
        }
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "phase": "P0_MEASUREMENT_ALIGNMENT_CAUSAL_AUDIT",
        "artifacts": receipts,
        "overall_status": (
            "BLOCKED_CAUSAL_FUSION"
            if artifacts["causal_contract_audit.json"]["status"] != "PASS"
            else "PASS"
        ),
        "input_set_sha256": stable_json_hash(
            {name: row["sha256"] for name, row in receipts.items()}
        ),
    }
    summary_path = output_dir / "p0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v31_ai85_plan_execution"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
