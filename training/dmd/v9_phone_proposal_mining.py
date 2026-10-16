"""Mine review-only phone proposals from frozen specialist and COCO detectors.

The input is limited to the 13 registered development false negatives. Outputs are
diagnostic proposals and annotated contact sheets; they are never labels, event
predictions, threshold-selection evidence, or production decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.ai.auxiliary import PoseEstimator, classify_phone_context
from backend.ai.detector import NormalizedDetection
from training.common import sha256_file
from training.data_paths import data_path
from training.dmd.v9_diagnostics import (
    DEVELOPMENT_SUBJECTS,
    PROTECTED_HOLDOUTS,
    assert_development_subject,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"
CONFIDENCE_FLOOR = 0.02
REVIEW_CONFIDENCE = 0.20
AGREEMENT_IOU = 0.25


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def intersection_over_union(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _best_pose(estimator: PoseEstimator, roi: np.ndarray) -> Any | None:
    return max(estimator.predict(roi), key=lambda item: item.confidence, default=None)


def _proposals(result: Any, source: str, pose: Any | None) -> list[dict[str, Any]]:
    rows = []
    for box in result.boxes:
        xyxy = [float(value) for value in box.xyxy[0].cpu().tolist()]
        confidence = float(box.conf.item())
        context, hand, face = classify_phone_context(
            NormalizedDetection(0, "phone", confidence, tuple(xyxy)), pose
        )
        rows.append(
            {
                "source": source,
                "confidence": confidence,
                "xyxy_in_driver_roi": xyxy,
                "geometry_context": context,
                "hand_proximity": float(hand),
                "face_proximity": float(face),
                "occupant_association": "UNVERIFIED_DRIVER_ROI_NO_TRACK_ID",
            }
        )
    return rows


def _annotate(roi: np.ndarray, proposals: list[dict[str, Any]], frame_index: int) -> np.ndarray:
    canvas = roi.copy()
    colors = {"specialist": (30, 210, 30), "coco_cell_phone": (255, 150, 20)}
    for proposal in proposals:
        x1, y1, x2, y2 = (int(value) for value in proposal["xyxy_in_driver_roi"])
        color = colors[proposal["source"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{proposal['proposal_id']} {proposal['confidence']:.2f} {proposal['geometry_context']}"
        )
        cv2.putText(
            canvas,
            label,
            (max(2, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"frame {frame_index} | REVIEW ONLY",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    target_height = 360
    resized = [
        cv2.resize(image, (round(image.shape[1] * target_height / image.shape[0]), target_height))
        for image in images
    ]
    return cv2.hconcat(resized)


def _render_report(output: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    text = f"""# V9 frozen phone proposal mining

This review-only experiment ran two unchanged checkpoints over
{summary["sampled_frames"]} original-video frames from the registered 13 development
false negatives.
The specialist detector produced proposals on {summary["specialist_frames_with_proposal"]} frames;
the independent COCO cell-phone class produced proposals on
{summary["auxiliary_frames_with_proposal"]} frames at the fixed {CONFIDENCE_FLOOR:.2f} audit floor.

- Cases with specialist proposals: {summary["cases_with_specialist_proposal"]}/13.
- Cases with auxiliary proposals: {summary["cases_with_auxiliary_proposal"]}/13.
- Cases with cross-model IoU >= {AGREEMENT_IOU:.2f}:
  {summary["cases_with_cross_model_agreement"]}/13.
- Cases with auxiliary confidence >= {REVIEW_CONFIDENCE:.2f} and hand/face proximity:
  {summary["cases_with_auxiliary_review_candidate"]}/13.
- Verified occupant tracks: 0/13.

These are proposal-coverage counts on FN-selected development data, not detector recall,
event recovery, or evidence for choosing an operating threshold. Recurrent cockpit objects can be
false cell-phone proposals. Human review must inspect the contact sheets and original video before
any box becomes a label.
No model weights, labels, event matcher, or production configuration changed.
"""
    (output / "11_PHONE_PROPOSAL_MINING.md").write_text(text, encoding="utf-8")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    trace_path = ROOT / "reports" / "ai85_next" / "detector_trace.json"
    diagnostics_path = output / "v9_diagnostics.json"
    trace = _read_json(trace_path)
    diagnostics = _read_json(diagnostics_path)
    subjects = {
        item["case_id"]: item["subject_id"]
        for item in diagnostics["phone_false_negatives"]["cases"]
    }
    if len(subjects) != 13 or set(subjects) != {row["case_id"] for row in trace["rows"]}:
        raise ValueError("proposal mining requires the frozen set of 13 V9 false negatives")
    if set(subjects.values()) != set(DEVELOPMENT_SUBJECTS):
        raise ValueError("unexpected V9 development subjects")
    if set(subjects.values()) & set(PROTECTED_HOLDOUTS):
        raise ValueError("protected holdout requested")

    specialist_path = ROOT / "models" / "active" / "v2" / "phone_detector.pt"
    auxiliary_path = ROOT / "models" / "auxiliary" / "yolo11n.pt"
    pose_path = ROOT / "models" / "auxiliary" / "yolo11n-pose.pt"
    specialist = YOLO(str(specialist_path))
    auxiliary = YOLO(str(auxiliary_path))
    pose_estimator = PoseEstimator(pose_path)
    external_root = data_path("external_dmd")
    review_dir = output / "phone_proposal_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    queue = []
    for trace_row in trace["rows"]:
        case_id = trace_row["case_id"]
        subject = subjects[case_id]
        assert_development_subject(subject)
        video = next((external_root / "subjects" / subject).glob("*_rgb_body.mp4"))
        capture = cv2.VideoCapture(str(video))
        samples = []
        annotated = []
        try:
            for sample in trace_row["samples"]:
                frame_index = int(sample["frame_index"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"decode failed: {case_id} frame {frame_index}")
                height, width = frame.shape[:2]
                bounds = [0, int(0.10 * height), int(0.65 * width), int(0.95 * height)]
                roi = frame[bounds[1] : bounds[3], bounds[0] : bounds[2]]
                pose = _best_pose(pose_estimator, roi)
                specialist_result = specialist.predict(
                    roi,
                    imgsz=640,
                    conf=CONFIDENCE_FLOOR,
                    iou=0.7,
                    device="cpu",
                    verbose=False,
                    max_det=10,
                )[0]
                auxiliary_result = auxiliary.predict(
                    roi,
                    imgsz=640,
                    conf=CONFIDENCE_FLOOR,
                    iou=0.7,
                    classes=[67],
                    device="cpu",
                    verbose=False,
                    max_det=10,
                )[0]
                specialist_rows = _proposals(specialist_result, "specialist", pose)
                auxiliary_rows = _proposals(auxiliary_result, "coco_cell_phone", pose)
                for index, proposal in enumerate(specialist_rows, start=1):
                    proposal["proposal_id"] = f"f{frame_index}-S{index}"
                for index, proposal in enumerate(auxiliary_rows, start=1):
                    proposal["proposal_id"] = f"f{frame_index}-C{index}"
                for proposal in specialist_rows:
                    proposal["nearest_cross_model_iou"] = max(
                        (
                            intersection_over_union(
                                proposal["xyxy_in_driver_roi"], other["xyxy_in_driver_roi"]
                            )
                            for other in auxiliary_rows
                        ),
                        default=0.0,
                    )
                for proposal in auxiliary_rows:
                    proposal["nearest_cross_model_iou"] = max(
                        (
                            intersection_over_union(
                                proposal["xyxy_in_driver_roi"], other["xyxy_in_driver_roi"]
                            )
                            for other in specialist_rows
                        ),
                        default=0.0,
                    )
                proposals = specialist_rows + auxiliary_rows
                samples.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(sample["timestamp_seconds"]),
                        "driver_roi_xyxy_in_full_frame": bounds,
                        "pose_available": pose is not None,
                        "proposals": proposals,
                    }
                )
                annotated.append(_annotate(roi, proposals, frame_index))
        finally:
            capture.release()
        sheet_path = review_dir / f"{case_id}.jpg"
        if not cv2.imwrite(str(sheet_path), _contact_sheet(annotated)):
            raise ValueError(f"could not write {sheet_path}")
        specialist_proposals = [
            proposal
            for sample in samples
            for proposal in sample["proposals"]
            if proposal["source"] == "specialist"
        ]
        auxiliary_proposals = [
            proposal
            for sample in samples
            for proposal in sample["proposals"]
            if proposal["source"] == "coco_cell_phone"
        ]
        cross_agreement = any(
            proposal["nearest_cross_model_iou"] >= AGREEMENT_IOU
            for proposal in specialist_proposals + auxiliary_proposals
        )
        auxiliary_review = any(
            proposal["confidence"] >= REVIEW_CONFIDENCE
            and proposal["geometry_context"] == "HANDHELD_USE"
            for proposal in auxiliary_proposals
        )
        row = {
            "case_id": case_id,
            "subject_id": subject,
            "samples": samples,
            "specialist_proposals": len(specialist_proposals),
            "auxiliary_proposals": len(auxiliary_proposals),
            "cross_model_agreement": cross_agreement,
            "auxiliary_review_candidate": auxiliary_review,
            "contact_sheet": sheet_path.relative_to(ROOT).as_posix(),
        }
        rows.append(row)
        queue.append(
            {
                "case_id": case_id,
                "specialist_proposals": len(specialist_proposals),
                "auxiliary_proposals": len(auxiliary_proposals),
                "cross_model_agreement": cross_agreement,
                "auxiliary_review_candidate": auxiliary_review,
                "contact_sheet": row["contact_sheet"],
                "human_box_decision": "",
                "human_notes": "",
            }
        )
        print(f"proposal mining completed: {case_id}", flush=True)

    payload = {
        "schema_version": 1,
        "scope": "FN-selected development proposal mining for human review only",
        "confidence_floor": CONFIDENCE_FLOOR,
        "review_confidence": REVIEW_CONFIDENCE,
        "agreement_iou": AGREEMENT_IOU,
        "models": {
            "specialist": {
                "path": specialist_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(specialist_path),
            },
            "coco_cell_phone": {
                "path": auxiliary_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(auxiliary_path),
                "class_id": 67,
            },
            "pose": {
                "path": pose_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(pose_path),
            },
        },
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "sampled_frames": sum(len(row["samples"]) for row in rows),
            "specialist_frames_with_proposal": sum(
                any(p["source"] == "specialist" for p in sample["proposals"])
                for row in rows
                for sample in row["samples"]
            ),
            "auxiliary_frames_with_proposal": sum(
                any(p["source"] == "coco_cell_phone" for p in sample["proposals"])
                for row in rows
                for sample in row["samples"]
            ),
            "cases_with_specialist_proposal": sum(row["specialist_proposals"] > 0 for row in rows),
            "cases_with_auxiliary_proposal": sum(row["auxiliary_proposals"] > 0 for row in rows),
            "cases_with_cross_model_agreement": sum(row["cross_model_agreement"] for row in rows),
            "cases_with_auxiliary_review_candidate": sum(
                row["auxiliary_review_candidate"] for row in rows
            ),
            "verified_occupant_tracks": 0,
        },
        "limitations": [
            "No phone bounding-box ground truth is available for these sampled frames.",
            "Proposal counts are not precision, recall, AP, or event recovery.",
            "The audit floor and review heuristic are not runtime operating thresholds.",
            "All proposals require original-video human review before labeling.",
        ],
        "ground_truth_changed": False,
        "event_matcher_changed": False,
        "fine_tuning_performed": False,
        "production_approved": False,
    }
    _write_json(output / "phone_proposal_mining.json", payload)
    with (output / "phone_proposal_review_queue.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    _render_report(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
