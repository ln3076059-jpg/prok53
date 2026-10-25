"""Replay fail-closed phone tracklets over saved autonomous candidate traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.ai.phone_tracker import PhoneEvidence, PhoneTrackletManager
from training.dmd.analyze_event_evidence import analyze_subject


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summarize_interval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tracklet_observations": len(rows),
        "states": dict(sorted(Counter(row["state"] for row in rows).items())),
        "verified_openers": sum(row["may_open_event"] for row in rows),
        "continuations": sum(row["state"] == "CONTINUATION" for row in rows),
        "new_rejected_continuations": sum(
            row["state"] == "CONTINUATION" and not row["was_admitted"]
            for row in rows
        ),
        "track_ids": sorted({row["track_id"] for row in rows}),
    }


def replay_subject(
    subject_dir: Path,
    *,
    frame_width: int = 1280,
    frame_height: int = 720,
    track_ttl_seconds: float = 2.0,
    continuation_ttl_seconds: float = 1.0,
) -> dict[str, Any]:
    observations = _read_jsonl(subject_dir / "autonomous_predictions.jsonl")
    candidates = _read_jsonl(subject_dir / "phone_candidate_records.jsonl")
    candidate_by_id = {
        str(row["candidate_id"]): row
        for row in candidates
        if row.get("candidate_id") is not None
    }
    manager = PhoneTrackletManager(
        track_ttl_seconds=track_ttl_seconds,
        continuation_ttl_seconds=continuation_ttl_seconds,
    )
    replay_rows: list[dict[str, Any]] = []
    for observation in observations:
        timestamp = float(observation["timestamp"])
        candidate_id = observation.get("candidate_id")
        if candidate_id is None or str(candidate_id) not in candidate_by_id:
            manager.advance(timestamp)
            continue
        candidate = candidate_by_id[str(candidate_id)]
        box = tuple(float(value) for value in candidate["box"])
        if len(box) != 4:
            continue
        associated = candidate.get("assoc_state") == "DRIVER_ASSOCIATED"
        handheld = max(
            float(candidate.get("hand_prox", 0.0)),
            float(candidate.get("face_prox", 0.0)),
        ) >= 0.40
        tracked = manager.update(
            PhoneEvidence(
                timestamp=timestamp,
                box=box,
                confidence=float(candidate.get("confidence", 0.0)),
                source=str(candidate.get("source", "UNKNOWN")),
                driver_associated=associated,
                handheld_verified=handheld,
            ),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        replay_rows.append({
            "timestamp": timestamp,
            "candidate_id": candidate_id,
            "track_id": tracked.track_id,
            "state": tracked.state.value,
            "may_open_event": tracked.may_open_event,
            "may_continue_event": tracked.may_continue_event,
            "was_admitted": float(observation.get("confidence", 0.0)) > 0.0,
            "rejection_reason": observation.get("rejection_reason"),
            "source": candidate.get("source"),
        })

    subject_analysis = analyze_subject(subject_dir)
    false_negative_replay = []
    for event in subject_analysis["ground_truth_events"]:
        if event["matched"]:
            continue
        start = float(event["ground_truth"]["start_seconds"])
        end = float(event["ground_truth"]["end_seconds"])
        rows = [row for row in replay_rows if start <= row["timestamp"] <= end]
        false_negative_replay.append({
            "ground_truth": event["ground_truth"],
            "duration_seconds": event["duration_seconds"],
            **_summarize_interval_rows(rows),
        })

    false_positive_replay = []
    for ledger in subject_analysis["false_positive_events"]:
        prediction = ledger["prediction"]
        start = float(prediction["start_seconds"])
        end = float(prediction["end_seconds"])
        rows = [row for row in replay_rows if start <= row["timestamp"] <= end]
        false_positive_replay.append({
            "reason": ledger["reason"],
            "prediction": prediction,
            **_summarize_interval_rows(rows),
        })

    return {
        "subject_dir": str(subject_dir),
        "configuration": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "track_ttl_seconds": track_ttl_seconds,
            "continuation_ttl_seconds": continuation_ttl_seconds,
            "verified_contract": "DRIVER_ASSOCIATED_AND_HAND_OR_FACE_PROXIMITY_GTE_0.40",
        },
        "selected_candidates": len(replay_rows),
        "state_counts": dict(sorted(Counter(row["state"] for row in replay_rows).items())),
        "new_rejected_continuations": sum(
            row["state"] == "CONTINUATION" and not row["was_admitted"]
            for row in replay_rows
        ),
        "false_negative_replay": false_negative_replay,
        "false_positive_replay": false_positive_replay,
    }


def replay_run(run_dir: Path) -> dict[str, Any]:
    aggregate = json.loads((run_dir / "event_matches.json").read_text(encoding="utf-8"))
    subjects = {
        subject_id: replay_subject(run_dir / "subjects" / subject_id)
        for subject_id in aggregate["cohort"]
    }
    return {
        "schema_version": "1.0.0",
        "scope": "OFFLINE_TRACKLET_REPLAY_NO_EVENT_METRICS",
        "run_dir": str(run_dir),
        "subjects": subjects,
        "total_new_rejected_continuations": sum(
            row["new_rejected_continuations"] for row in subjects.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "total_new_rejected_continuations": result["total_new_rejected_continuations"],
        "subjects": {
            subject: row["state_counts"] for subject, row in result["subjects"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
