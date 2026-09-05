from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from training.identity_contract import validate_identity_contract


def generate_annotation_skeleton(
    video_id: str,
    vehicle_track_id: int | None = None,
    cabin_epoch: int = 0,
    occupant_track_ids: list[int] | None = None,
    fps: float = 30.0,
    frame_count: int = 300,
    source_id: str = "camera-1",
    is_provided_cabin: bool = False,
) -> dict:
    """Generate a deterministic annotation skeleton with locked IDs from runtime tracking.

    Ensures human annotators annotate on the exact deterministic namespaces required by
    the evaluation pipeline. Leaves events empty and review status as AI_REVIEWED_PROPOSAL.
    """
    if is_provided_cabin:
        vehicle_id = f"video:{video_id}:provided-vehicle"
        cabin_id = f"video:{video_id}:provided-cabin"
    else:
        v_track = 1 if vehicle_track_id is None else vehicle_track_id
        vehicle_id = f"video:{video_id}:vehicle-track:{v_track}"
        cabin_id = f"{vehicle_id}:cabin:{cabin_epoch}"

    tracks = occupant_track_ids or [1]
    occupants = []
    context_intervals = []

    for idx, track_id in enumerate(tracks):
        occupant_id = f"{cabin_id}:occupant-track:{track_id}"
        occupants.append(
            {
                "occupant_id": occupant_id,
                "role": "unknown",
                "role_confidence": 0.0,
                "reviewer_confirmed_role": False,
            }
        )
        context_intervals.append(
            {
                "context_id": f"ctx-{video_id}-occ-{track_id}-full",
                "occupant_id": occupant_id,
                "start_frame": 0,
                "end_frame": frame_count,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "NO_PHONE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
            }
        )

        # Validate that each generated identity strictly obeys the contract
        contract_errors = validate_identity_contract(video_id, vehicle_id, cabin_id, occupant_id)
        if contract_errors:
            raise ValueError(f"Generated identity contract violation: {contract_errors}")

    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    skeleton = {
        "sequence_id": f"seq-{video_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "video_id": video_id,
        "vehicle_id": vehicle_id,
        "cabin_id": cabin_id,
        "source_id": source_id,
        "fps": fps,
        "frame_count": frame_count,
        "start_time": now_iso,
        "end_time": now_iso,
        "occupants": occupants,
        "events": [],
        "context_intervals": context_intervals,
        "context": {
            "inside_vehicle": True,
            "outside_vehicle_person": False,
            "motorcycle_flag": False,
        },
        "review_provenance": {
            "reviewer_type": "AI",
            "status": "AI_REVIEWED_PROPOSAL",
            "annotation_version": "v2.0",
        },
    }
    return skeleton


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract locked identity manifest and generate annotation skeleton"
    )
    parser.add_argument("--video-id", required=True, help="Video identifier")
    parser.add_argument("--vehicle-track-id", type=int, default=1, help="Vehicle track ID")
    parser.add_argument("--cabin-epoch", type=int, default=0, help="Cabin epoch")
    parser.add_argument(
        "--occupant-tracks",
        type=int,
        nargs="+",
        default=[1],
        help="Occupant track IDs detected in upper body",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video frames per second")
    parser.add_argument("--frame-count", type=int, default=300, help="Total frame count")
    parser.add_argument("--source-id", default="camera-1", help="Camera source identifier")
    parser.add_argument(
        "--provided-cabin",
        action="store_true",
        help="Use provided-cabin instead of vehicle-track cabin",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")

    args = parser.parse_args()
    skeleton = generate_annotation_skeleton(
        video_id=args.video_id,
        vehicle_track_id=args.vehicle_track_id,
        cabin_epoch=args.cabin_epoch,
        occupant_track_ids=args.occupant_tracks,
        fps=args.fps,
        frame_count=args.frame_count,
        source_id=args.source_id,
        is_provided_cabin=args.provided_cabin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(skeleton, handle, indent=2)
        handle.write("\n")
    print(f"Generated locked identity annotation skeleton at {args.output}")


if __name__ == "__main__":
    main()
