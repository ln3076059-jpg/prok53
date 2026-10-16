"""Roadwatch V9 diagnostics over frozen development evidence.

This module never changes labels, event matching, model weights, or protected holdouts.
It turns the existing V8 traces into explicit phone-box, occupant, geometry, image-quality,
and review-readiness evidence before any new fine-tuning is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.ai.auxiliary import PoseEstimator, classify_phone_context
from backend.ai.detector import NormalizedDetection
from training.common import sha256_file
from training.data_paths import data_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"
DEVELOPMENT_SUBJECTS = ("gB-9", "gZ-36", "gC-14")
PROTECTED_HOLDOUTS = ("gZ-37", "gE-28")
ACTION_GATE = 0.80


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def assert_development_subject(subject: str) -> None:
    if subject in PROTECTED_HOLDOUTS or subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError(f"subject is not registered for V9 development diagnostics: {subject}")


def summarize_detection_row(row: dict[str, Any]) -> dict[str, Any]:
    samples = [sample for sample in row["samples"] if sample.get("decode_status") == "OK"]
    proposal_samples = [sample for sample in samples if sample.get("phone_proposals")]
    proposals = [proposal for sample in proposal_samples for proposal in sample["phone_proposals"]]
    associations = [str(item.get("occupant_association", "MISSING")) for item in proposals]
    tracked = [item for item in associations if "NO_TRACK_ID" not in item and item != "MISSING"]
    if not proposals:
        box_status = "MISSING_ALL_SAMPLED_FRAMES"
        occupant_status = "BLOCKED_NO_PHONE_BOX"
        geometry_status = "BLOCKED_NO_PHONE_BOX"
    else:
        box_status = (
            "AVAILABLE_ON_SUBSET" if len(proposal_samples) < len(samples) else "AVAILABLE_ALL"
        )
        occupant_status = "AVAILABLE" if len(tracked) == len(proposals) else "MISSING_TRACK_ID"
        geometry_status = (
            "READY_FOR_LANDMARK_CHECK"
            if occupant_status == "AVAILABLE"
            else "UNVERIFIED_ROI_LEVEL_ASSOCIATION"
        )
    return {
        "decoded_samples": len(samples),
        "samples_with_phone_box": len(proposal_samples),
        "phone_box_coverage": len(proposal_samples) / len(samples) if samples else 0.0,
        "phone_proposals": len(proposals),
        "maximum_phone_confidence": max(
            (float(item["confidence"]) for item in proposals), default=None
        ),
        "phone_box_status": box_status,
        "occupant_id_status": occupant_status,
        "geometry_status": geometry_status,
        "occupant_association_values": sorted(set(associations)),
    }


def classify_fn_bottleneck(
    detection: dict[str, Any], baseline_phone_probability: float | None
) -> str:
    if detection["phone_proposals"] == 0:
        return "PHONE_DETECTOR_NO_PROPOSAL"
    if baseline_phone_probability is None or baseline_phone_probability < ACTION_GATE:
        return "PHONE_BOX_PRESENT_ACTION_UNDER_GATE"
    if detection["occupant_id_status"] != "AVAILABLE":
        return "PHONE_AND_ACTION_PRESENT_OCCUPANT_UNVERIFIED"
    return "TEMPORAL_OR_FUSION_GATE"


def _driver_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[int(0.10 * height) : int(0.95 * height), : int(0.65 * width)]


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index} from {video}")
    return _driver_roi(frame)


def _quality(center: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    motion = (
        np.mean(cv2.absdiff(gray, before_gray), dtype=np.float64)
        + np.mean(cv2.absdiff(after_gray, gray), dtype=np.float64)
    ) / (2.0 * 255.0)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray < 24)),
        "highlight_fraction": float(np.mean(gray > 245)),
        "three_frame_motion_mean": float(motion),
    }


def _best_pose(poses: list[Any]) -> Any | None:
    return max(poses, key=lambda pose: pose.confidence, default=None)


def _proposal_geometry(
    pose_estimator: PoseEstimator,
    video: Path,
    detector_row: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        (sample, proposal)
        for sample in detector_row["samples"]
        for proposal in sample.get("phone_proposals", [])
    ]
    if not candidates:
        return {"status": "BLOCKED_NO_PHONE_BOX"}
    sample, proposal = max(candidates, key=lambda item: float(item[1]["confidence"]))
    roi = _read_frame(video, int(sample["frame_index"]))
    pose = _best_pose(pose_estimator.predict(roi))
    if pose is None:
        return {
            "status": "BLOCKED_NO_POSE",
            "frame_index": sample["frame_index"],
            "phone_confidence": proposal["confidence"],
        }
    box = tuple(float(value) for value in proposal["xyxy_in_driver_roi"])
    context, hand, face = classify_phone_context(
        NormalizedDetection(0, "phone", float(proposal["confidence"]), box), pose
    )
    return {
        "status": "UNVERIFIED_NO_OCCUPANT_TRACK",
        "frame_index": sample["frame_index"],
        "phone_confidence": float(proposal["confidence"]),
        "pose_confidence": float(pose.confidence),
        "hand_proximity": float(hand),
        "face_proximity": float(face),
        "geometry_context": context,
    }


def _seatbelt_audit() -> dict[str, Any]:
    queue_path = ROOT / "reports" / "v7_optimization" / "seatbelt_uncertain_review_queue.csv"
    with queue_path.open(encoding="utf-8-sig", newline="") as stream:
        queue = list(csv.DictReader(stream))
    pending_path = data_path("manifests/v2_seatbelt_uncertain_ui_review.json")
    pending = _read_json(pending_path)
    results = _read_json(ROOT / "reports" / "ai85_plus" / "seatbelt_results.json")
    variants = []
    for variant in results["variants"]:
        raw = variant["evaluation"]["raw"]
        variants.append(
            {
                "id": variant["id"],
                "macro_f1": raw["macro_f1"],
                "balanced_accuracy": raw["balanced_accuracy"],
                "uncertain_recall": raw["per_class"]["uncertain_or_occluded"]["recall"],
                "false_unfastened_rate": variant["evaluation"]["false_unfastened"][
                    "rate_among_true_fastened"
                ],
                "abstention_rate": variant["evaluation"]["abstention"]["output_uncertain_rate"],
            }
        )
    return {
        "frozen_variants": variants,
        "best_development_macro_f1": max(item["macro_f1"] for item in variants),
        "legacy_uncertain_error_queue": {
            "samples": len(queue),
            "roi_available_locally": sum(row["roi_available_locally"] == "True" for row in queue),
            "correct": sum(row["model_correct"] == "True" for row in queue),
            "high_confidence_wrong_at_0_90": sum(
                row["model_correct"] == "False" and float(row["top1_probability"]) >= 0.90
                for row in queue
            ),
            "status": "HUMAN_REVIEW_REQUIRED",
        },
        "pending_uncertain_review_manifest": {
            "path": pending_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(pending_path),
            "samples": len(pending),
            "review_status": dict(Counter(str(row["review_status"]) for row in pending)),
            "automatic_training_labels_present": sum(
                row.get("automatic_training_label") is not None for row in pending
            ),
            "occupant_roles": dict(Counter(str(row["occupant_role"]) for row in pending)),
            "unique_effective_groups": len({str(row["effective_group_id"]) for row in pending}),
        },
        "fine_tuning_status": "BLOCKED_PENDING_HUMAN_APPROVAL",
        "reason": "No reviewed three-state training labels; invisible belt must remain uncertain.",
    }


def run(output: Path = DEFAULT_OUTPUT, run_pose: bool = True) -> dict[str, Any]:
    evidence_path = ROOT / "reports" / "ai85_plus" / "evidence_diagnostics.json"
    detector_path = ROOT / "reports" / "ai85_next" / "detector_trace.json"
    manifest_path = ROOT / "reports" / "ai85_upgrade" / "fn_frame_manifest.json"
    comparison_path = ROOT / "reports" / "ai85_plus" / "metrics_comparison.json"
    representation_path = ROOT / "reports" / "ai85_next" / "analysis.json"
    runtime_path = ROOT / "reports" / "ai85_plus" / "runtime_benchmark.json"
    gpu_path = (
        ROOT / "reports" / "ai85_plus" / "kaggle_gpu" / "version1" / "gpu_frozen_results.json"
    )

    evidence = _read_json(evidence_path)
    detector = _read_json(detector_path)
    manifest = _read_json(manifest_path)
    comparison = _read_json(comparison_path)
    representation = _read_json(representation_path)
    detector_by_case = {row["case_id"]: row for row in detector["rows"]}
    manifest_by_case = {row["case_id"]: row for row in manifest}
    case_ids = {case["case_id"] for case in evidence["false_negative_cases"]}
    if (
        len(case_ids) != 13
        or case_ids != set(detector_by_case)
        or case_ids != set(manifest_by_case)
    ):
        raise ValueError(
            "V9 requires the same 13 FN cases across evidence, detector, and ROI manifests"
        )

    external_root = data_path("external_dmd")
    pose_estimator = (
        PoseEstimator(ROOT / "models" / "auxiliary" / "yolo11n-pose.pt") if run_pose else None
    )
    cases = []
    for case in evidence["false_negative_cases"]:
        subject = case["subject_id"]
        assert_development_subject(subject)
        video = next((external_root / "subjects" / subject).glob("*_rgb_body.mp4"))
        row = detector_by_case[case["case_id"]]
        manifest_row = manifest_by_case[case["case_id"]]
        frame_index = int(manifest_row["frame_index"])
        center = _read_frame(video, frame_index)
        before = _read_frame(video, frame_index - 3)
        after = _read_frame(video, frame_index + 3)
        detection = summarize_detection_row(row)
        baseline_probability = case["sampling"]["T0"]["raw_phone_probability_max"]
        dense_probability = case["sampling"]["T1"]["raw_phone_probability_max"]
        pose = _best_pose(pose_estimator.predict(center)) if pose_estimator else None
        geometry = (
            _proposal_geometry(pose_estimator, video, row)
            if pose_estimator
            else {"status": "NOT_RUN"}
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "subject_id": subject,
                "action": case["truth"]["source_action"],
                "duration_seconds": case["truth"]["duration_seconds"],
                "visual_review": case["visual_review"],
                "visual_review_scope": case["visual_review_scope"],
                "roi_source": manifest_row["path"],
                "roi_sha256": manifest_row["sha256"],
                "quality": _quality(center, before, after),
                "pose": {
                    "status": "AVAILABLE" if pose else "MISSING",
                    "confidence": float(pose.confidence) if pose else None,
                    "face_points": len(pose.face_points) if pose else 0,
                    "hand_points": len(pose.hand_points) if pose else 0,
                },
                "detector": detection,
                "proposal_geometry": geometry,
                "action_probability": {
                    "T0_max": baseline_probability,
                    "T1_dense_max": dense_probability,
                    "fixed_gate": ACTION_GATE,
                    "T0_reaches_gate": bool(
                        baseline_probability is not None and baseline_probability >= ACTION_GATE
                    ),
                    "T1_reaches_gate": bool(
                        dense_probability is not None and dense_probability >= ACTION_GATE
                    ),
                },
                "root_cause": classify_fn_bottleneck(detection, baseline_probability),
                "human_review_required": True,
            }
        )

    counts = Counter(item["root_cause"] for item in cases)
    decoded = sum(item["detector"]["decoded_samples"] for item in cases)
    proposal_samples = sum(item["detector"]["samples_with_phone_box"] for item in cases)
    subject_silhouette = float(representation["subject_bias"]["silhouette_cosine_by_subject"])
    action_silhouette = float(representation["subject_bias"]["silhouette_cosine_by_action"])
    event_rows = comparison["event_metrics"]
    baseline = next(row for row in event_rows if row["candidate"] == "AI85 next controlled binary")
    fusion_rows = []
    for row in event_rows:
        fusion_rows.append(
            {
                **row,
                "delta_tp": row["true_positives"] - baseline["true_positives"],
                "delta_fp": row["false_positives"] - baseline["false_positives"],
                "delta_fn": row["false_negatives"] - baseline["false_negatives"],
            }
        )

    payload = {
        "schema_version": 1,
        "scope": "development-only frozen diagnostics; no training or policy search",
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "protected_holdouts": list(PROTECTED_HOLDOUTS),
        "protected_holdouts_accessed": False,
        "ground_truth_changed": False,
        "event_matcher_changed": False,
        "phone_false_negatives": {
            "cases": cases,
            "summary": {
                "cases": len(cases),
                "root_causes": dict(counts),
                "decoded_detector_samples": decoded,
                "samples_with_phone_box": proposal_samples,
                "sampled_phone_box_coverage": proposal_samples / decoded if decoded else 0.0,
                "cases_with_any_phone_box": sum(
                    item["detector"]["phone_proposals"] > 0 for item in cases
                ),
                "cases_with_verified_occupant_id": sum(
                    item["detector"]["occupant_id_status"] == "AVAILABLE" for item in cases
                ),
                "T0_cases_reaching_action_gate": sum(
                    item["action_probability"]["T0_reaches_gate"] for item in cases
                ),
                "T1_cases_reaching_action_gate": sum(
                    item["action_probability"]["T1_reaches_gate"] for item in cases
                ),
                "T1_cases_matched": sum(
                    case["sampling"]["T1"]["matched"] for case in evidence["false_negative_cases"]
                ),
            },
        },
        "representation": {
            "frozen_embedding_subject_silhouette": subject_silhouette,
            "frozen_embedding_action_silhouette": action_silhouette,
            "subject_to_action_separability_ratio": (
                subject_silhouette / action_silhouette
                if not math.isclose(action_silhouette, 0.0)
                else None
            ),
            "interpretation": representation["subject_bias"]["interpretation"],
            "dense_sampling_result": "One long event recovered; short-event matching remained 0/4.",
            "clip_feature_comparison_status": "BLOCKED_NO_FROZEN_CLIP_ARTIFACT_IN_REPOSITORY",
        },
        "fusion_ablation": fusion_rows,
        "seatbelt": _seatbelt_audit(),
        "runtime": {
            "local": _read_json(runtime_path),
            "kaggle_gpu": _read_json(gpu_path),
            "comparison_warning": (
                "Different inputs and scopes; no controlled CPU/GPU speedup claim."
            ),
        },
        "source_artifacts": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                evidence_path,
                detector_path,
                manifest_path,
                comparison_path,
                representation_path,
                runtime_path,
                gpu_path,
            )
        },
        "fine_tuning_performed": False,
        "production_approved": False,
        "independent_status": "INDEPENDENT_AI85_NOT_ESTABLISHED",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "v9_diagnostics.json", payload)
    with (output / "phone_fn_human_review_queue.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "case_id",
            "subject_id",
            "action",
            "duration_seconds",
            "root_cause",
            "phone_box_status",
            "occupant_id_status",
            "visual_review",
            "roi_source",
            "human_decision",
            "human_notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in cases:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "subject_id": item["subject_id"],
                    "action": item["action"],
                    "duration_seconds": item["duration_seconds"],
                    "root_cause": item["root_cause"],
                    "phone_box_status": item["detector"]["phone_box_status"],
                    "occupant_id_status": item["detector"]["occupant_id_status"],
                    "visual_review": item["visual_review"],
                    "roi_source": item["roi_source"],
                    "human_decision": "",
                    "human_notes": "",
                }
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args()
    result = run(args.output, run_pose=not args.skip_pose)
    print(json.dumps(result["phone_false_negatives"]["summary"], indent=2))


if __name__ == "__main__":
    main()
