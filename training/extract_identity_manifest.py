from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from backend.ai.events import EventCandidate
from training.common import sha256_file
from training.identity_contract import validate_identity_contract


def extract_identities_from_predictions_csv(predictions_csv_path: Path) -> dict[str, Any]:
    """Extract proven runtime entity identities from an evaluator-compatible predictions CSV.

    Validates that each entity adheres to canonical deterministic namespaces and
    tracks the observation duration and evidence hash.
    """
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"predictions CSV not found: {predictions_csv_path}")

    source_sha256 = sha256_file(predictions_csv_path)

    with predictions_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("predictions CSV is empty; cannot extract runtime identity manifest")

    videos: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "video_id": "",
            "fps": 30.0,
            "max_seconds": 0.0,
            "vehicle_ids": set(),
            "cabin_ids": set(),
            "occupants": {},
        }
    )

    proven_set: set[tuple[str, str, str, str]] = set()

    for idx, row in enumerate(rows, start=2):
        video_id = row.get("video_id", "").strip()
        vehicle_id = row.get("vehicle_id", "").strip()
        cabin_id = row.get("cabin_id", "").strip()
        occupant_id = row.get("occupant_id", "").strip()

        if not (video_id and vehicle_id and cabin_id and occupant_id):
            continue

        contract_errors = validate_identity_contract(video_id, vehicle_id, cabin_id, occupant_id)
        if contract_errors:
            raise ValueError(f"row {idx}: invalid runtime identity contract: {contract_errors}")

        proven_set.add((video_id, vehicle_id, cabin_id, occupant_id))

        try:
            start_sec = float(row.get("start_seconds", "0"))
            end_sec = float(row.get("end_seconds", "0"))
        except ValueError:
            start_sec, end_sec = 0.0, 0.0

        obs_count = 1
        try:
            obs_count = int(row.get("observation_count", 1))
        except ValueError:
            obs_count = 1

        v_data = videos[video_id]
        v_data["video_id"] = video_id
        v_data["max_seconds"] = max(v_data["max_seconds"], end_sec)
        v_data["vehicle_ids"].add(vehicle_id)
        v_data["cabin_ids"].add(cabin_id)

        occ_map = v_data["occupants"]
        if occupant_id not in occ_map:
            # Parse track id from occupant_id if present
            track_num = 1
            if ":occupant-track:" in occupant_id:
                try:
                    track_num = int(occupant_id.split(":occupant-track:")[-1])
                except ValueError:
                    track_num = 1
            occ_map[occupant_id] = {
                "occupant_id": occupant_id,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_track_id": track_num,
                "first_seconds": start_sec,
                "last_seconds": end_sec,
                "observation_count": obs_count,
            }
        else:
            rec = occ_map[occupant_id]
            rec["first_seconds"] = min(rec["first_seconds"], start_sec)
            rec["last_seconds"] = max(rec["last_seconds"], end_sec)
            rec["observation_count"] += obs_count

    videos_out: dict[str, Any] = {}
    for vid, v_data in videos.items():
        fps = 30.0
        frame_count = max(1, int(math.ceil(v_data["max_seconds"] * fps)))
        videos_out[vid] = {
            "video_id": vid,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": v_data["max_seconds"],
            "vehicle_ids": sorted(list(v_data["vehicle_ids"])),
            "cabin_ids": sorted(list(v_data["cabin_ids"])),
            "occupants": sorted(list(v_data["occupants"].values()), key=lambda x: x["occupant_id"]),
        }

    proven_list = [
        {"video_id": vid, "vehicle_id": veh, "cabin_id": cab, "occupant_id": occ}
        for (vid, veh, cab, occ) in sorted(proven_set)
    ]

    return {
        "manifest_version": "v2.0",
        "source_type": "RUNTIME_PREDICTIONS_CSV",
        "source_sha256": source_sha256,
        "source_path": str(predictions_csv_path.resolve()),
        "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "videos": videos_out,
        "proven_identities": proven_list,
    }


def extract_identities_from_candidates(
    candidates: Iterable[EventCandidate],
    video_id: str,
    fps: float = 30.0,
    frame_count: int = 300,
    source_sha256: str = "",
) -> dict[str, Any]:
    """Extract proven runtime entity identities from in-memory EventCandidate objects."""
    vehicle_ids: set[str] = set()
    cabin_ids: set[str] = set()
    occupants: dict[str, dict[str, Any]] = {}
    proven_set: set[tuple[str, str, str, str]] = set()

    for cand in candidates:
        veh_id = cand.vehicle_context_id
        cab_id = cand.cabin_id
        occ_id = cand.occupant_id

        if not (veh_id and cab_id and occ_id):
            continue

        contract_errors = validate_identity_contract(video_id, veh_id, cab_id, occ_id)
        if contract_errors:
            raise ValueError(f"invalid runtime identity contract on candidate: {contract_errors}")

        vehicle_ids.add(veh_id)
        cabin_ids.add(cab_id)
        proven_set.add((video_id, veh_id, cab_id, occ_id))

        if occ_id not in occupants:
            track_num = 1
            if ":occupant-track:" in occ_id:
                try:
                    track_num = int(occ_id.split(":occupant-track:")[-1])
                except ValueError:
                    track_num = 1
            occupants[occ_id] = {
                "occupant_id": occ_id,
                "vehicle_id": veh_id,
                "cabin_id": cab_id,
                "occupant_track_id": track_num,
                "first_seconds": cand.start_timestamp,
                "last_seconds": cand.end_timestamp,
                "observation_count": cand.observation_count,
            }
        else:
            rec = occupants[occ_id]
            rec["first_seconds"] = min(rec["first_seconds"], cand.start_timestamp)
            rec["last_seconds"] = max(rec["last_seconds"], cand.end_timestamp)
            rec["observation_count"] += cand.observation_count

    videos_out = {
        video_id: {
            "video_id": video_id,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "vehicle_ids": sorted(list(vehicle_ids)),
            "cabin_ids": sorted(list(cabin_ids)),
            "occupants": sorted(list(occupants.values()), key=lambda x: x["occupant_id"]),
        }
    }

    proven_list = [
        {"video_id": vid, "vehicle_id": veh, "cabin_id": cab, "occupant_id": occ}
        for (vid, veh, cab, occ) in sorted(proven_set)
    ]

    return {
        "manifest_version": "v2.0",
        "source_type": "RUNTIME_EVENT_CANDIDATES",
        "source_sha256": source_sha256,
        "source_path": f"memory:candidates:{video_id}",
        "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "videos": videos_out,
        "proven_identities": proven_list,
    }


def freeze_identity_manifest(manifest_path: Path, output_lock_path: Path) -> dict[str, Any]:
    """Freeze and cryptographically lock an extracted runtime identity manifest."""
    if output_lock_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen identity manifest: {output_lock_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "v2.0":
        raise ValueError("identity manifest version must be v2.0")

    proven = manifest.get("proven_identities", [])
    if not proven:
        raise ValueError("identity manifest declares no proven identities")

    errors: list[str] = []
    video_ids: set[str] = set()
    for item in proven:
        vid = item.get("video_id", "")
        veh_id = item.get("vehicle_id", "")
        cab_id = item.get("cabin_id", "")
        occ_id = item.get("occupant_id", "")
        video_ids.add(vid)
        for err in validate_identity_contract(vid, veh_id, cab_id, occ_id):
            errors.append(err)

    if errors:
        raise ValueError("cannot freeze identity manifest:\n- " + "\n- ".join(errors))

    manifest_sha = sha256_file(manifest_path)
    lock_data = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": manifest_sha,
        "manifest_file": manifest_path.name,
        "source_type": manifest.get("source_type", "UNKNOWN"),
        "source_sha256": manifest.get("source_sha256", ""),
        "video_ids": sorted(list(video_ids)),
        "proven_identities": proven,
        "identity_count": len(proven),
        "locked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    output_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock_path.open("w", encoding="utf-8") as handle:
        json.dump(lock_data, handle, indent=2)
        handle.write("\n")

    return lock_data


def generate_annotation_skeleton(
    manifest: dict[str, Any],
    video_id: str,
    source_id: str = "camera-1",
) -> dict[str, Any]:
    """Generate an annotation skeleton strictly from proven identities in a manifest.

    Ensures:
    1. Only proven runtime IDs from the manifest are present.
    2. start_time < end_time (duration matches frame_count / fps).
    3. Business ground truth (phone_state, seatbelt_state, visibility, conditions)
       is NOT fabricated with safe defaults, but explicitly set to UNKNOWN / UNREVIEWED.
    4. Review status is AI_REVIEWED_PROPOSAL (not human approved).
    """
    videos = manifest.get("videos", {})
    if video_id not in videos:
        raise ValueError(f"video_id '{video_id}' not found in identity manifest")

    v_info = videos[video_id]
    vehicle_ids = v_info.get("vehicle_ids", [])
    cabin_ids = v_info.get("cabin_ids", [])
    occupants_proven = v_info.get("occupants", [])

    if not vehicle_ids or not cabin_ids or not occupants_proven:
        raise ValueError(f"video_id '{video_id}' has incomplete entity definitions in manifest")

    vehicle_id = vehicle_ids[0]
    cabin_id = cabin_ids[0]
    fps = float(v_info.get("fps", 30.0))
    frame_count = int(v_info.get("frame_count", 300))
    duration_sec = frame_count / fps if fps > 0 else 1.0

    start_dt = datetime.now(UTC)
    end_dt = start_dt + timedelta(seconds=duration_sec)
    start_time_iso = start_dt.isoformat().replace("+00:00", "Z")
    end_time_iso = end_dt.isoformat().replace("+00:00", "Z")

    occupants = []
    context_intervals = []

    for idx, occ in enumerate(occupants_proven, start=1):
        occ_id = occ["occupant_id"]
        occupants.append(
            {
                "occupant_id": occ_id,
                "role": "unknown",
                "role_confidence": 0.0,
                "reviewer_confirmed_role": False,
            }
        )
        # Explicitly unreviewed interval covering the full sequence
        context_intervals.append(
            {
                "context_id": f"ctx-{video_id}-occ-{idx}-unreviewed",
                "occupant_id": occ_id,
                "start_frame": 0,
                "end_frame": frame_count,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "UNKNOWN",
                "seatbelt_state": "UNCERTAIN_OR_OCCLUDED",
                "visibility": "UNREVIEWED",
                "conditions": "UNREVIEWED",
                "notes": "SKELETON_UNREVIEWED_IDENTITY_PROVENANCE",
            }
        )

    skeleton = {
        "sequence_id": f"seq-{video_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "video_id": video_id,
        "vehicle_id": vehicle_id,
        "cabin_id": cabin_id,
        "source_id": source_id,
        "fps": fps,
        "frame_count": frame_count,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
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
            "evidence_hash": manifest.get("source_sha256", ""),
            "identity_manifest_sha256": manifest.get("manifest_sha256", ""),
        },
    }
    return skeleton


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract locked identity manifest from runtime tracking evidence"
    )
    parser.add_argument("--predictions-csv", type=Path, help="Path to runtime predictions CSV")
    parser.add_argument("--tracking-json", type=Path, help="Path to runtime tracking JSON artifact")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--freeze-lock", type=Path, help="Optional output path to freeze manifest into lock JSON")
    parser.add_argument("--skeleton-output", type=Path, help="Optional output path to generate annotation skeleton")
    parser.add_argument("--video-id", help="Video ID for skeleton generation")

    args = parser.parse_args()

    if args.predictions_csv:
        manifest = extract_identities_from_predictions_csv(args.predictions_csv)
    elif args.tracking_json:
        data = json.loads(args.tracking_json.read_text(encoding="utf-8"))
        manifest = data
    else:
        parser.error("must provide either --predictions-csv or --tracking-json")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Extracted identity manifest saved to {args.output_manifest}")

    if args.freeze_lock:
        freeze_identity_manifest(args.output_manifest, args.freeze_lock)
        print(f"Frozen identity manifest lock saved to {args.freeze_lock}")

    if args.skeleton_output:
        vid = args.video_id or next(iter(manifest.get("videos", {}).keys()))
        skeleton = generate_annotation_skeleton(manifest, vid)
        args.skeleton_output.parent.mkdir(parents=True, exist_ok=True)
        with args.skeleton_output.open("w", encoding="utf-8") as handle:
            json.dump(skeleton, handle, indent=2)
            handle.write("\n")
        print(f"Annotation skeleton saved to {args.skeleton_output}")


if __name__ == "__main__":
    main()
