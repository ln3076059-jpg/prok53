from __future__ import annotations

import argparse
import csv
import hashlib
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


def extract_identity_manifest_from_track_records(
    records: list[dict[str, Any]],
    source_path: str = "",
    source_sha256: str = "",
) -> dict[str, Any]:
    """Extract proven runtime entity identities from a list of track records.

    Each record must describe a tracked occupant observation window:
    - video_id: str
    - vehicle_id: str
    - cabin_id: str
    - occupant_id: str
    - fps: float
    - frame_count: int
    - duration: float (or duration_seconds)
    - first_frame: int
    - last_frame: int
    - tracking_evidence: dict
    """
    videos: dict[str, dict[str, Any]] = {}
    proven_set: set[tuple[str, str, str, str]] = set()

    for idx, rec in enumerate(records):
        video_id = str(rec.get("video_id", "")).strip()
        vehicle_id = str(rec.get("vehicle_id", "")).strip()
        cabin_id = str(rec.get("cabin_id", "")).strip()
        occupant_id = str(rec.get("occupant_id", "")).strip()

        if not (video_id and vehicle_id and cabin_id and occupant_id):
            continue

        contract_errors = validate_identity_contract(video_id, vehicle_id, cabin_id, occupant_id)
        if contract_errors:
            raise ValueError(f"track {idx}: invalid runtime identity contract: {contract_errors}")

        proven_set.add((video_id, vehicle_id, cabin_id, occupant_id))

        fps = float(rec.get("fps", 30.0))
        frame_count = int(rec.get("frame_count", 0))
        duration = float(rec.get("duration", rec.get("duration_seconds", frame_count / fps if fps > 0 else 0.0)))
        video_sha256 = str(rec.get("video_sha256", "")).strip()

        if video_id not in videos:
            videos[video_id] = {
                "video_id": video_id,
                "video_sha256": video_sha256,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "vehicle_ids": set(),
                "cabin_ids": set(),
                "occupants": {},
            }
        else:
            v_entry = videos[video_id]
            v_entry["fps"] = fps
            v_entry["frame_count"] = max(v_entry["frame_count"], frame_count)
            v_entry["duration_seconds"] = max(v_entry["duration_seconds"], duration)
            if video_sha256 and not v_entry["video_sha256"]:
                v_entry["video_sha256"] = video_sha256

        v_data = videos[video_id]
        v_data["vehicle_ids"].add(vehicle_id)
        v_data["cabin_ids"].add(cabin_id)

        first_frame = int(rec.get("first_frame", 0))
        last_frame = int(rec.get("last_frame", 0))
        evidence = rec.get("tracking_evidence", {})
        first_sec = float(evidence.get("first_seconds", first_frame / fps if fps > 0 else 0.0))
        last_sec = float(evidence.get("last_seconds", last_frame / fps if fps > 0 else 0.0))
        obs_count = int(evidence.get("observation_count", max(1, last_frame - first_frame + 1)))

        track_num = 1
        if ":occupant-track:" in occupant_id:
            try:
                track_num = int(occupant_id.split(":occupant-track:")[-1])
            except ValueError:
                track_num = 1

        occ_map = v_data["occupants"]
        if occupant_id not in occ_map:
            occ_map[occupant_id] = {
                "occupant_id": occupant_id,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_track_id": track_num,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "first_seconds": first_sec,
                "last_seconds": last_sec,
                "observation_count": obs_count,
                "average_confidence": float(evidence.get("average_confidence", 1.0)),
                "assigned_role": str(evidence.get("assigned_role", "unknown")),
            }
        else:
            existing = occ_map[occupant_id]
            existing["first_frame"] = min(existing["first_frame"], first_frame)
            existing["last_frame"] = max(existing["last_frame"], last_frame)
            existing["first_seconds"] = min(existing["first_seconds"], first_sec)
            existing["last_seconds"] = max(existing["last_seconds"], last_sec)
            existing["observation_count"] += obs_count

    videos_out: dict[str, Any] = {}
    for vid, v_data in sorted(videos.items()):
        videos_out[vid] = {
            "video_id": vid,
            "video_sha256": v_data.get("video_sha256", ""),
            "fps": v_data["fps"],
            "frame_count": v_data["frame_count"],
            "duration_seconds": v_data["duration_seconds"],
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
        "source_type": "RUNTIME_IDENTITY_TRACKS",
        "evaluation_scope": "CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING",
        "eligible_for_frozen_event_evaluation": True,
        "source_sha256": source_sha256,
        "source_path": source_path,
        "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "videos": videos_out,
        "proven_identities": proven_list,
    }


def extract_identity_manifest_from_tracks(tracks_path: Path) -> dict[str, Any]:
    """Extract proven runtime entity identities from runtime_identity_tracks artifact (.jsonl or .json)."""
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks artifact not found: {tracks_path}")

    source_sha256 = sha256_file(tracks_path)
    records: list[dict[str, Any]] = []

    content = tracks_path.read_text(encoding="utf-8").strip()
    if not content:
        records = []
    elif content.startswith("["):
        records = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return extract_identity_manifest_from_track_records(
        records,
        source_path=str(tracks_path.resolve()),
        source_sha256=source_sha256,
    )


def extract_identities_from_predictions_csv(predictions_csv_path: Path) -> dict[str, Any]:
    """Extract proven runtime entity identities from an evaluator-compatible predictions CSV.

    Provided for backward-compatibility. For rigorous event evaluation, prefer
    extract_identity_manifest_from_tracks to capture full tracking without event-filtering bias.
    """
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"predictions CSV not found: {predictions_csv_path}")

    source_sha256 = sha256_file(predictions_csv_path)

    with predictions_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {
            "manifest_version": "v2.0",
            "source_type": "RUNTIME_PREDICTIONS_CSV",
            "source_sha256": source_sha256,
            "source_path": str(predictions_csv_path.resolve()),
            "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "videos": {},
            "proven_identities": [],
        }

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
        "evaluation_scope": "LEGACY_DEBUG_ONLY",
        "eligible_for_frozen_event_evaluation": False,
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
            "duration_seconds": frame_count / fps if fps > 0 else 0.0,
            "vehicle_ids": sorted(list(vehicle_ids)),
            "cabin_ids": sorted(list(cabin_ids)),
            "occupants": sorted(list(occupants.values()), key=lambda x: x["occupant_id"]),
        }
    }

    proven_list = [
        {"video_id": vid, "vehicle_id": veh, "cabin_id": cab, "occupant_id": occ}
        for (vid, veh, cab, occ) in sorted(proven_set)
    ]

    effective_sha = source_sha256
    if not effective_sha:
        digest = hashlib.sha256(f"{video_id}:{fps}:{frame_count}:{len(proven_list)}".encode("utf-8"))
        effective_sha = digest.hexdigest()

    return {
        "manifest_version": "v2.0",
        "source_type": "RUNTIME_EVENT_CANDIDATES",
        "evaluation_scope": "LEGACY_DEBUG_ONLY",
        "eligible_for_frozen_event_evaluation": False,
        "source_sha256": effective_sha,
        "source_path": f"memory:candidates:{video_id}",
        "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "videos": videos_out,
        "proven_identities": proven_list,
    }


def extract_identity_manifest_from_annotations(
    annotation_paths: list[Path],
    source_manifest_path: Path | None = None,
    video_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract an independent ground-truth identity manifest from sequence annotations.

    Guarantees:
    1. Identity roster is independent of whether the model-under-test detects occupants.
    2. Eligible for final frozen event evaluation (source_type = INDEPENDENT_GROUND_TRUTH_ANNOTATIONS).
    3. Scope is FULL_SYSTEM_EVENT_EVALUATION (allows measuring false negatives for missed occupants).
    4. Validates identity contracts for every declared occupant.
    5. Preserves full video provenance (fps, frame_count, duration, video_sha256).
    """
    if not annotation_paths:
        raise ValueError("no annotation paths provided for identity manifest extraction")

    videos: dict[str, dict[str, Any]] = {}
    proven_set: set[tuple[str, str, str, str]] = set()

    for p in sorted(annotation_paths):
        if not p.is_file():
            raise ValueError(f"annotation file not found: {p}")
        ann = json.loads(p.read_text(encoding="utf-8"))
        vid = str(ann.get("video_id", "")).strip()
        if not vid:
            raise ValueError(f"{p}: sequence annotation missing video_id")
        veh_id = str(ann.get("vehicle_id", "")).strip()
        cab_id = str(ann.get("cabin_id", "")).strip()
        fps = float(ann.get("fps", 30.0))
        frame_count = int(ann.get("frame_count", 0))
        duration = float(ann.get("duration_seconds", frame_count / fps if fps > 0 else 0.0))

        video_sha256 = str(ann.get("video_sha256", "")).strip()
        if video_metadata and vid in video_metadata:
            vm = video_metadata[vid]
            video_sha256 = str(vm.get("sha256", video_sha256)).strip()
            fps = float(vm.get("fps", fps))
            frame_count = int(vm.get("frame_count", frame_count))
            duration = float(vm.get("duration_seconds", duration))

        if vid not in videos:
            videos[vid] = {
                "video_id": vid,
                "video_sha256": video_sha256,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "vehicle_ids": set(),
                "cabin_ids": set(),
                "occupants": {},
            }
        else:
            v_data = videos[vid]
            v_data["fps"] = fps
            v_data["frame_count"] = max(v_data["frame_count"], frame_count)
            v_data["duration_seconds"] = max(v_data["duration_seconds"], duration)
            if video_sha256 and not v_data["video_sha256"]:
                v_data["video_sha256"] = video_sha256

        v_data = videos[vid]
        if veh_id:
            v_data["vehicle_ids"].add(veh_id)
        if cab_id:
            v_data["cabin_ids"].add(cab_id)

        occupants = ann.get("occupants", [])
        for occ in occupants:
            occ_id = str(occ.get("occupant_id", "")).strip()
            role = str(occ.get("role", "unknown")).strip()
            contract_errors = validate_identity_contract(vid, veh_id, cab_id, occ_id)
            if contract_errors:
                raise ValueError(f"{p}: invalid identity contract for {occ_id}: {contract_errors}")

            proven_set.add((vid, veh_id, cab_id, occ_id))

            track_num = 1
            if ":occupant-track:" in occ_id:
                try:
                    track_num = int(occ_id.split(":occupant-track:")[-1])
                except ValueError:
                    track_num = 1

            if occ_id not in v_data["occupants"]:
                v_data["occupants"][occ_id] = {
                    "occupant_id": occ_id,
                    "vehicle_id": veh_id,
                    "cabin_id": cab_id,
                    "occupant_track_id": track_num,
                    "first_frame": 0,
                    "last_frame": frame_count,
                    "first_seconds": 0.0,
                    "last_seconds": duration,
                    "observation_count": frame_count,
                    "average_confidence": 1.0,
                    "assigned_role": role,
                    "source": "INDEPENDENT_GROUND_TRUTH_ANNOTATION",
                }

    videos_out: dict[str, Any] = {}
    for vid, v in videos.items():
        videos_out[vid] = {
            "video_id": vid,
            "video_sha256": v["video_sha256"],
            "fps": v["fps"],
            "frame_count": v["frame_count"],
            "duration_seconds": v["duration_seconds"],
            "vehicle_ids": sorted(list(v["vehicle_ids"])),
            "cabin_ids": sorted(list(v["cabin_ids"])),
            "occupants": sorted(list(v["occupants"].values()), key=lambda x: x["occupant_id"]),
        }

    proven_list = [
        {"video_id": item[0], "vehicle_id": item[1], "cabin_id": item[2], "occupant_id": item[3]}
        for item in sorted(list(proven_set))
    ]

    if source_manifest_path is not None and source_manifest_path.is_file():
        source_path_str = str(source_manifest_path.resolve())
        source_sha = sha256_file(source_manifest_path)
    elif len(annotation_paths) == 1:
        source_path_str = str(annotation_paths[0].resolve())
        source_sha = sha256_file(annotation_paths[0])
    else:
        source_path_str = str(annotation_paths[0].parent.resolve())
        hasher = hashlib.sha256()
        for p in sorted(annotation_paths):
            hasher.update(p.read_bytes())
        source_sha = hasher.hexdigest()

    return {
        "manifest_version": "v2.0",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_type": "INDEPENDENT_GROUND_TRUTH_ANNOTATIONS",
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "eligible_for_frozen_event_evaluation": True,
        "source_path": source_path_str,
        "source_sha256": source_sha,
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

    governed_frozen_source_types = {
        "RUNTIME_IDENTITY_TRACKS",
        "INDEPENDENT_GROUND_TRUTH_ANNOTATIONS",
        "INDEPENDENT_IDENTITY_ROSTER",
    }
    legacy_debug_source_types = {
        "RUNTIME_PREDICTIONS_CSV",
        "RUNTIME_EVENT_CANDIDATES",
    }
    allowed_source_types = governed_frozen_source_types | legacy_debug_source_types

    source_type = manifest.get("source_type")
    if source_type not in allowed_source_types:
        raise ValueError(f"unrecognized identity manifest source_type: {source_type!r}")

    source_sha = manifest.get("source_sha256", "")
    if not source_sha or len(source_sha) != 64:
        raise ValueError("identity manifest requires a valid 64-character source_sha256")

    source_path_str = manifest.get("source_path", "")
    if not source_path_str:
        raise ValueError("identity manifest requires a non-empty source_path")

    if source_type in governed_frozen_source_types:
        eligible_for_frozen_event_evaluation = True
        evaluation_scope = manifest.get(
            "evaluation_scope",
            "CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING"
            if source_type == "RUNTIME_IDENTITY_TRACKS"
            else "FULL_SYSTEM_EVENT_EVALUATION",
        )
        if source_path_str.startswith("memory:"):
            raise ValueError(
                f"governed source_type '{source_type}' requires an immutable disk artifact, not {source_path_str}"
            )
        source_p = Path(source_path_str)
        if not source_p.exists():
            raise ValueError(f"identity manifest source_path does not exist on disk: {source_path_str}")
        if source_p.is_file():
            disk_sha = sha256_file(source_p)
            if disk_sha != source_sha:
                raise ValueError(
                    f"identity manifest source_sha256 mismatch: recorded {source_sha} != disk {disk_sha}"
                )
    else:
        eligible_for_frozen_event_evaluation = False
        evaluation_scope = "LEGACY_DEBUG_ONLY"
        if not source_path_str.startswith("memory:"):
            source_p = Path(source_path_str)
            if not source_p.is_file():
                raise ValueError(f"identity manifest source_path does not exist on disk: {source_path_str}")
            disk_sha = sha256_file(source_p)
            if disk_sha != source_sha:
                raise ValueError(
                    f"identity manifest source_sha256 mismatch: recorded {source_sha} != disk {disk_sha}"
                )

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

    lock_videos: dict[str, Any] = {}
    for vid, v_entry in manifest.get("videos", {}).items():
        lock_videos[vid] = {
            "sha256": str(v_entry.get("video_sha256", "")).strip(),
            "fps": float(v_entry.get("fps", 30.0)),
            "frame_count": int(v_entry.get("frame_count", 0)),
            "duration_seconds": float(v_entry.get("duration_seconds", 0.0)),
        }

    manifest_sha = sha256_file(manifest_path)
    lock_data = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": manifest_sha,
        "manifest_file": manifest_path.name,
        "source_type": source_type,
        "source_path": source_path_str,
        "source_sha256": source_sha,
        "eligible_for_frozen_event_evaluation": eligible_for_frozen_event_evaluation,
        "evaluation_scope": evaluation_scope,
        "video_ids": sorted(list(video_ids)),
        "videos": lock_videos,
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
    cabin_id: str | None = None,
    manifest_lock: dict[str, Any] | None = None,
    source_id: str = "camera-1",
) -> dict[str, Any]:
    """Generate an annotation skeleton strictly from proven identities in a manifest.

    Guarantees:
    1. Only proven runtime IDs from the manifest are present.
    2. Partitioned cleanly per (video_id, vehicle_id, cabin_id) to support multi-cabin scenes.
    3. start_time < end_time (duration matches frame_count / fps).
    4. Business ground truth is NOT fabricated:
       - phone_state = "UNKNOWN", seatbelt_state = "UNCERTAIN_OR_OCCLUDED"
       - visibility = "UNREVIEWED", conditions = "UNREVIEWED"
       - inside_vehicle = None, outside_vehicle_person = None, motorcycle_flag = None (null in JSON)
    5. Copies manifest_lock["manifest_sha256"] directly into review_provenance["identity_manifest_sha256"].
    6. Review status is AI_REVIEWED_PROPOSAL (not human approved).
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

    if cabin_id is not None:
        if cabin_id not in cabin_ids:
            raise ValueError(f"cabin_id '{cabin_id}' not found in manifest for video '{video_id}'")
        selected_cabin_id = cabin_id
    else:
        selected_cabin_id = cabin_ids[0]

    # Resolve corresponding vehicle_id
    if ":cabin:" in selected_cabin_id:
        selected_vehicle_id = selected_cabin_id.rsplit(":cabin:", 1)[0]
    elif selected_cabin_id == f"video:{video_id}:provided-cabin":
        selected_vehicle_id = f"video:{video_id}:provided-vehicle"
    else:
        selected_vehicle_id = vehicle_ids[0]

    # Partition occupants strictly to selected cabin
    cabin_occupants = [occ for occ in occupants_proven if occ.get("cabin_id") == selected_cabin_id]
    if not cabin_occupants:
        cabin_occupants = occupants_proven

    fps = float(v_info.get("fps", 30.0))
    frame_count = int(v_info.get("frame_count", 300))
    duration_sec = frame_count / fps if fps > 0 else 1.0

    start_dt = datetime.now(UTC)
    end_dt = start_dt + timedelta(seconds=duration_sec)
    start_time_iso = start_dt.isoformat().replace("+00:00", "Z")
    end_time_iso = end_dt.isoformat().replace("+00:00", "Z")

    occupants = []
    context_intervals = []

    for idx, occ in enumerate(cabin_occupants, start=1):
        occ_id = occ["occupant_id"]
        occupants.append(
            {
                "occupant_id": occ_id,
                "role": "unknown",
                "role_confidence": 0.0,
                "reviewer_confirmed_role": False,
            }
        )
        # Explicitly unreviewed proposal interval covering full sequence
        # Safety context fields are None (null in JSON) - NOT fabricated!
        context_intervals.append(
            {
                "context_id": f"ctx-{video_id}-occ-{idx}-unreviewed",
                "occupant_id": occ_id,
                "start_frame": 0,
                "end_frame": frame_count,
                "inside_vehicle": None,
                "outside_vehicle_person": None,
                "motorcycle_flag": None,
                "phone_state": "UNKNOWN",
                "seatbelt_state": "UNCERTAIN_OR_OCCLUDED",
                "visibility": "UNREVIEWED",
                "conditions": "UNREVIEWED",
                "notes": "SKELETON_UNREVIEWED_IDENTITY_PROVENANCE",
            }
        )

    manifest_sha = ""
    if manifest_lock is not None:
        manifest_sha = manifest_lock.get("manifest_sha256", "")
    if not manifest_sha:
        manifest_sha = manifest.get("manifest_sha256", "")

    skeleton = {
        "sequence_id": f"seq-{video_id}-{selected_cabin_id.replace(':', '-')}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "video_id": video_id,
        "vehicle_id": selected_vehicle_id,
        "cabin_id": selected_cabin_id,
        "source_id": source_id,
        "fps": fps,
        "frame_count": frame_count,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "occupants": occupants,
        "events": [],
        "context_intervals": context_intervals,
        "context": {
            "inside_vehicle": None,
            "outside_vehicle_person": None,
            "motorcycle_flag": None,
        },
        "review_provenance": {
            "reviewer_type": "AI",
            "status": "AI_REVIEWED_PROPOSAL",
            "annotation_version": "v2.0",
            "evidence_hash": manifest.get("source_sha256", ""),
            "identity_manifest_sha256": manifest_sha,
        },
    }
    return skeleton


def generate_annotation_skeletons_for_video(
    manifest: dict[str, Any],
    video_id: str,
    manifest_lock: dict[str, Any] | None = None,
    source_id: str = "camera-1",
) -> list[dict[str, Any]]:
    """Generate separate sequence annotation skeletons for each cabin in a video."""
    videos = manifest.get("videos", {})
    if video_id not in videos:
        raise ValueError(f"video_id '{video_id}' not found in identity manifest")
    v_info = videos[video_id]
    cabin_ids = v_info.get("cabin_ids", [])
    return [
        generate_annotation_skeleton(
            manifest,
            video_id,
            cabin_id=cid,
            manifest_lock=manifest_lock,
            source_id=source_id,
        )
        for cid in cabin_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract locked identity manifest from runtime tracking evidence or independent ground truth"
    )
    parser.add_argument("--tracks", "--tracks-jsonl", type=Path, dest="tracks", help="Path to runtime_identity_tracks artifact (.jsonl or .json)")
    parser.add_argument("--annotations", nargs="+", type=Path, help="Paths to human-approved sequence annotation JSON files")
    parser.add_argument("--annotations-dir", type=Path, help="Directory containing human-approved sequence annotation JSON files")
    parser.add_argument("--predictions-csv", type=Path, help="Legacy debug: Path to runtime predictions CSV")
    parser.add_argument("--tracking-json", type=Path, help="Legacy debug: Path to runtime tracking JSON")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--freeze-lock", type=Path, help="Optional output path to freeze manifest into lock JSON")
    parser.add_argument("--skeleton-output", type=Path, help="Optional output path to generate annotation skeleton")
    parser.add_argument("--video-id", help="Video ID for skeleton generation")
    parser.add_argument("--cabin-id", help="Optional Cabin ID for skeleton generation")

    args = parser.parse_args()

    if args.tracks:
        manifest = extract_identity_manifest_from_tracks(args.tracks)
    elif args.annotations or args.annotations_dir:
        ann_paths: list[Path] = []
        if args.annotations:
            ann_paths.extend(args.annotations)
        if args.annotations_dir:
            ann_paths.extend(sorted(args.annotations_dir.glob("*.json")))
        manifest = extract_identity_manifest_from_annotations(ann_paths)
    elif args.predictions_csv:
        manifest = extract_identities_from_predictions_csv(args.predictions_csv)
    elif args.tracking_json:
        data = json.loads(args.tracking_json.read_text(encoding="utf-8"))
        if isinstance(data, list):
            manifest = extract_identity_manifest_from_track_records(
                data,
                source_path=str(args.tracking_json.resolve()),
                source_sha256=sha256_file(args.tracking_json),
            )
        else:
            manifest = data
    else:
        parser.error("must provide either --tracks, --annotations/--annotations-dir, --predictions-csv, or --tracking-json")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Extracted identity manifest saved to {args.output_manifest}")

    lock_data = None
    if args.freeze_lock:
        lock_data = freeze_identity_manifest(args.output_manifest, args.freeze_lock)
        print(f"Frozen identity manifest lock saved to {args.freeze_lock}")

    if args.skeleton_output:
        vid = args.video_id or next(iter(manifest.get("videos", {}).keys()))
        skeleton = generate_annotation_skeleton(
            manifest,
            vid,
            cabin_id=args.cabin_id,
            manifest_lock=lock_data,
        )
        args.skeleton_output.parent.mkdir(parents=True, exist_ok=True)
        with args.skeleton_output.open("w", encoding="utf-8") as handle:
            json.dump(skeleton, handle, indent=2)
            handle.write("\n")
        print(f"Annotation skeleton saved to {args.skeleton_output}")


if __name__ == "__main__":
    main()
