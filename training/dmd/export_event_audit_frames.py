"""Export review-only FN/FP contact sheets from an autonomous DMD run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from training.dmd.analyze_event_evidence import analyze_run


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_SUBJECTS = {"gC-14", "gZ-36"}
SOURCE_COLORS = {
    "specialist_roi": (60, 210, 60),
    "auxiliary_roi": (255, 170, 30),
}


def _sample_times(
    start: float,
    end: float,
    candidate_rows: list[dict[str, Any]],
    limit: int = 4,
) -> list[float]:
    duration = max(end - start, 0.0)
    anchors = [start + duration * ratio for ratio in (0.1, 0.5, 0.9)]
    candidates = sorted(
        candidate_rows,
        key=lambda row: float(row.get("confidence", 0.0)),
        reverse=True,
    )
    ordered = [float(row["timestamp"]) for row in candidates] + anchors
    unique: list[float] = []
    for timestamp in ordered:
        timestamp = min(max(timestamp, start), end)
        if all(abs(timestamp - existing) > 0.05 for existing in unique):
            unique.append(timestamp)
        if len(unique) == limit:
            break
    return sorted(unique)


def _contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("Contact sheet needs at least one image")
    target_width = 640
    resized = [
        cv2.resize(
            image,
            (target_width, round(image.shape[0] * target_width / image.shape[1])),
        )
        for image in images
    ]
    target_height = max(image.shape[0] for image in resized)
    padded = []
    for image in resized:
        if image.shape[0] < target_height:
            image = cv2.copyMakeBorder(
                image,
                0,
                target_height - image.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )
        padded.append(image)
    if len(padded) % 2:
        padded.append(np.zeros_like(padded[0]))
    rows = [cv2.hconcat(padded[index : index + 2]) for index in range(0, len(padded), 2)]
    return cv2.vconcat(rows)


def _video_for_subject(subject_id: str) -> Path:
    if subject_id not in DEVELOPMENT_SUBJECTS:
        raise ValueError(f"Audit is limited to development subjects: {subject_id}")
    matches = sorted(
        (ROOT / "dataset" / "external_dmd" / "subjects" / subject_id).glob(
            "*_rgb_body.mp4"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one BODY video for {subject_id}, found {len(matches)}")
    return matches[0]


def _render_frame(
    capture: cv2.VideoCapture,
    timestamp: float,
    candidates: list[dict[str, Any]],
    case_label: str,
) -> tuple[np.ndarray, float]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = max(0, round(timestamp * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Could not decode frame {frame_index} at {timestamp:.3f}s")
    decoded_time = frame_index / fps
    nearby = [
        row for row in candidates if abs(float(row["timestamp"]) - decoded_time) <= 0.3
    ]
    for row in nearby:
        box = row.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (round(float(value)) for value in box)
        source = str(row.get("source", "UNKNOWN"))
        color = SOURCE_COLORS.get(source, (220, 220, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{source} {float(row.get('confidence', 0.0)):.2f} {row.get('assoc_state', '')}"
        cv2.putText(
            frame,
            label,
            (max(5, x1), max(24, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{case_label} | t={decoded_time:.3f}s | proposals={len(nearby)} | REVIEW ONLY",
        (12, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, decoded_time


def export_audit(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    report = analyze_run(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []
    for subject_id, subject_report in report["subjects"].items():
        video_path = _video_for_subject(subject_id)
        candidate_path = run_dir / "subjects" / subject_id / "phone_candidate_records.jsonl"
        candidates = [
            json.loads(line)
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cases: list[tuple[str, float, float, dict[str, Any]]] = []
        for index, event in enumerate(subject_report["ground_truth_events"], start=1):
            if event["matched"]:
                continue
            ground_truth = event["ground_truth"]
            cases.append((
                f"FN-{index:02d}",
                float(ground_truth["start_seconds"]),
                float(ground_truth["end_seconds"]),
                event,
            ))
        for index, ledger in enumerate(subject_report["false_positive_events"], start=1):
            prediction = ledger["prediction"]
            cases.append((
                f"FP-{index:02d}",
                float(prediction["start_seconds"]),
                float(prediction["end_seconds"]),
                ledger,
            ))

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open {video_path}")
        try:
            for case_id, start, end, payload in cases:
                interval_candidates = [
                    row for row in candidates if start <= float(row["timestamp"]) <= end
                ]
                timestamps = _sample_times(start, end, interval_candidates)
                label = f"{subject_id} {case_id} {start:.2f}-{end:.2f}s"
                rendered = [
                    _render_frame(capture, timestamp, candidates, label)
                    for timestamp in timestamps
                ]
                sheet_path = output_dir / f"{subject_id}_{case_id}.jpg"
                if not cv2.imwrite(
                    str(sheet_path),
                    _contact_sheet([frame for frame, _ in rendered]),
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                ):
                    raise ValueError(f"Could not write {sheet_path}")
                manifest_cases.append({
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "kind": case_id.split("-", 1)[0],
                    "candidate_count": len(interval_candidates),
                    "sampled_timestamps": [round(value, 3) for value in timestamps],
                    "decoded_timestamps": [round(value, 3) for _, value in rendered],
                    "contact_sheet": sheet_path.resolve().relative_to(ROOT.resolve()).as_posix(),
                    "source_payload": payload,
                    "human_review": {
                        "phone_state": "",
                        "driver_association": "",
                        "notes": "",
                    },
                })
        finally:
            capture.release()

    result = {
        "schema_version": "1.0.0",
        "scope": "DEVELOPMENT_FN_FP_VISUAL_REVIEW_ONLY",
        "run_dir": str(run_dir),
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = export_audit(args.run_dir, args.output_dir)
    print(json.dumps({"case_count": result["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
