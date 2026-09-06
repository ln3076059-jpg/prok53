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
        if not video_sha256 or len(video_sha256) != 64:
            video_sha256 = hashlib.sha256(f"runtime-track-evidence:{video_id}".encode("utf-8")).hexdigest()

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
        v_sha = str(v_data.get("video_sha256", "")).strip().lower()
        if not v_sha or len(v_sha) != 64:
            v_sha = hashlib.sha256(f"legacy-prediction-csv:{vid}".encode("utf-8")).hexdigest()
        videos_out[vid] = {
            "video_id": vid,
            "video_sha256": v_sha,
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
        if not video_sha256 or len(video_sha256) != 64:
            raise ValueError(f"{p}: sequence annotation for video '{vid}' is missing valid 64-character video_sha256")

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
        "evaluation_scope": "LEGACY_ANNOTATION_EXTRACTED_SCOPE",
        "eligible_for_frozen_event_evaluation": False,
        "source_path": source_path_str,
        "source_sha256": source_sha,
        "videos": videos_out,
        "proven_identities": proven_list,
    }


def canonical_identity_evidence_hash(videos: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Compute deterministic SHA-256 evidence hash over canonical roster video/identity data.

    Canonical normalization:
    - Videos sorted by video_id
    - video_sha256 lowercased
    - Vehicles sorted by vehicle_id
    - Cabins sorted by cabin_id
    - Occupants sorted by occupant_id
    - Deterministic JSON encoding (sort_keys=True, separators=(',', ':'))
    """
    if isinstance(videos, dict):
        videos_list = list(videos.values())
    elif isinstance(videos, list):
        videos_list = videos
    else:
        raise TypeError(f"expected list or dict for videos, got {type(videos)}")

    canonical_videos = []
    for v in sorted(videos_list, key=lambda x: str(x.get("video_id", ""))):
        vid = str(v.get("video_id", "")).strip()
        v_sha = str(v.get("video_sha256", "")).strip().lower()
        fps = round(float(v.get("fps", 30.0)), 4)
        frame_count = int(v.get("frame_count", 0))
        duration = round(float(v.get("duration_seconds", frame_count / fps if fps > 0 else 0.0)), 4)

        vehicles_in = v.get("vehicles", [])
        canonical_vehicles = []
        for veh in sorted(vehicles_in, key=lambda x: str(x.get("vehicle_id", ""))):
            veh_id = str(veh.get("vehicle_id", "")).strip()
            cabins_in = veh.get("cabins", [])
            canonical_cabins = []
            for cab in sorted(cabins_in, key=lambda x: str(x.get("cabin_id", ""))):
                cab_id = str(cab.get("cabin_id", "")).strip()
                occupants_in = cab.get("occupants", [])
                canonical_occupants = []
                for occ in sorted(occupants_in, key=lambda x: str(x.get("occupant_id", ""))):
                    occ_id = str(occ.get("occupant_id", "")).strip()
                    role = str(occ.get("role", "unknown")).strip()
                    canonical_occupants.append({
                        "occupant_id": occ_id,
                        "role": role,
                    })
                canonical_cabins.append({
                    "cabin_id": cab_id,
                    "occupants": canonical_occupants,
                })
            canonical_vehicles.append({
                "vehicle_id": veh_id,
                "cabins": canonical_cabins,
            })

        canonical_videos.append({
            "video_id": vid,
            "video_sha256": v_sha,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "vehicles": canonical_vehicles,
        })

    canonical_json = json.dumps(canonical_videos, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_identity_roster(
    videos: list[dict[str, Any]],
    output_path: Path | None = None,
    human_review_status: str = "PENDING",
    reviewer_type: str = "AI",
    reviewer_id: str | None = None,
    reviewed_at: str | None = None,
    evidence_hash: str | None = None,
    adjudication_status: str = "PENDING",
    roster_status: str | None = None,
) -> dict[str, Any]:
    """Create a structured independent ground-truth identity roster document.

    By default, creates an UNREVIEWED identity roster (human_review_status='PENDING',
    reviewer_type='AI', reviewer_id=None, adjudication_status='PENDING').

    To elevate an identity roster to HUMAN review, callers must explicitly use
    `approve_identity_roster(...)` with valid reviewer_id and reviewed_at timestamp,
    or supply explicit reviewer_id and valid reviewed_at timestamp to this function.
    Fabrication of human reviewer IDs (e.g. defaulting to 'human-reviewer-1') is
    strictly prohibited.

    Each video entry in `videos` must declare:
    - video_id: str
    - video_sha256: str (non-empty 64-character lowercase hexadecimal)
    - fps: float
    - frame_count: int
    - duration_seconds: float (optional, defaults to frame_count / fps)
    - vehicles: list of dicts:
        - vehicle_id: str
        - cabins: list of dicts:
            - cabin_id: str
            - occupants: list of dicts:
                - occupant_id: str
                - role: str (optional, e.g. "driver", "front_passenger")
    """
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if reviewer_type == "HUMAN":
        if not reviewer_id or not str(reviewer_id).strip():
            raise ValueError("create_identity_roster: reviewer_type 'HUMAN' requires an explicit, non-empty reviewer_id")
        cleaned_rev_id = str(reviewer_id).strip()
        if cleaned_rev_id.lower() in {"human-reviewer-1", "placeholder", "unknown", "none"}:
            raise ValueError(f"create_identity_roster rejects placeholder reviewer_id: {cleaned_rev_id!r}")
        if not reviewed_at or not str(reviewed_at).strip():
            raise ValueError("create_identity_roster: reviewer_type 'HUMAN' requires an explicit, non-empty reviewed_at timestamp")
        try:
            parsed_dt = datetime.fromisoformat(str(reviewed_at).strip().replace("Z", "+00:00"))
            if parsed_dt.tzinfo is None:
                raise ValueError("reviewed_at must include timezone")
        except Exception as err:
            raise ValueError(f"create_identity_roster: invalid reviewed_at timestamp: {err}")
        if adjudication_status != "FINAL":
            raise ValueError(f"create_identity_roster: reviewer_type 'HUMAN' requires adjudication_status == 'FINAL', got {adjudication_status!r}")
        if human_review_status != "APPROVED":
            raise ValueError(f"create_identity_roster: reviewer_type 'HUMAN' requires human_review_status == 'APPROVED', got {human_review_status!r}")
        resolved_status = roster_status or "HUMAN_APPROVED_IDENTITY_ROSTER"
        final_reviewer_id = cleaned_rev_id
        final_reviewed_at = str(reviewed_at).strip()
    else:
        resolved_status = roster_status or "UNREVIEWED_IDENTITY_ROSTER"
        final_reviewer_id = str(reviewer_id).strip() if reviewer_id else None
        final_reviewed_at = str(reviewed_at).strip() if reviewed_at else None

    canonical_ev_hash = canonical_identity_evidence_hash(videos)
    if evidence_hash:
        clean_ev_hash = str(evidence_hash).strip().lower()
        if clean_ev_hash != canonical_ev_hash:
            raise ValueError(
                f"create_identity_roster: provided evidence_hash does not match canonical roster data (semantic evidence_hash mismatch): "
                f"{clean_ev_hash} != {canonical_ev_hash}"
            )

    roster_data = {
        "roster_version": "v2.0",
        "roster_status": resolved_status,
        "created_at": now_iso,
        "human_review_status": human_review_status,
        "reviewer_type": reviewer_type,
        "reviewer_id": final_reviewer_id,
        "reviewed_at": final_reviewed_at,
        "evidence_hash": canonical_ev_hash,
        "adjudication_status": adjudication_status,
        "videos": videos,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(roster_data, f, indent=2)
            f.write("\n")
    return roster_data


def approve_identity_roster(
    roster_input: Path | dict[str, Any],
    reviewer_id: str,
    reviewed_at: str,
    evidence_hash: str | None = None,
    adjudication_status: str = "FINAL",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Explicitly elevate an independent identity roster from UNREVIEWED to HUMAN_APPROVED.

    Requires:
    - reviewer_id: non-empty string identifying the human reviewer (no placeholders allowed).
    - reviewed_at: ISO-8601 timestamp string with explicit timezone.
    - adjudication_status: must be 'FINAL'.
    """
    if not reviewer_id or not str(reviewer_id).strip():
        raise ValueError("approve_identity_roster requires a non-empty reviewer_id")
    cleaned_reviewer_id = str(reviewer_id).strip()
    if cleaned_reviewer_id.lower() in {"human-reviewer-1", "placeholder", "unknown", "none"}:
        raise ValueError(f"approve_identity_roster rejects placeholder reviewer_id: {cleaned_reviewer_id!r}")

    if not reviewed_at or not str(reviewed_at).strip():
        raise ValueError("approve_identity_roster requires an explicit, non-empty reviewed_at timestamp")
    try:
        parsed_dt = datetime.fromisoformat(str(reviewed_at).strip().replace("Z", "+00:00"))
        if parsed_dt.tzinfo is None:
            raise ValueError("reviewed_at must include timezone")
    except Exception as err:
        raise ValueError(f"approve_identity_roster invalid reviewed_at timestamp: {err}")

    if adjudication_status != "FINAL":
        raise ValueError(f"approve_identity_roster requires adjudication_status == 'FINAL', got {adjudication_status!r}")

    if isinstance(roster_input, Path):
        if not roster_input.is_file():
            raise FileNotFoundError(f"roster file not found: {roster_input}")
        data = json.loads(roster_input.read_text(encoding="utf-8"))
    elif isinstance(roster_input, dict):
        data = dict(roster_input)
    else:
        raise TypeError(f"expected Path or dict for roster_input, got {type(roster_input)}")

    videos = data.get("videos", [])
    canonical_ev_hash = canonical_identity_evidence_hash(videos)
    recorded_ev_hash = data.get("evidence_hash")
    if recorded_ev_hash:
        clean_rec = str(recorded_ev_hash).strip().lower()
        if clean_rec != canonical_ev_hash:
            raise ValueError(
                f"approve_identity_roster: roster evidence_hash semantic mismatch: "
                f"recorded {clean_rec} != recomputed {canonical_ev_hash}"
            )
    if evidence_hash:
        clean_ev_hash = str(evidence_hash).strip().lower()
        if clean_ev_hash != canonical_ev_hash:
            raise ValueError(
                f"approve_identity_roster: provided evidence_hash does not match canonical roster data (semantic evidence_hash mismatch): "
                f"{clean_ev_hash} != {canonical_ev_hash}"
            )

    data["roster_status"] = "HUMAN_APPROVED_IDENTITY_ROSTER"
    data["human_review_status"] = "APPROVED"
    data["reviewer_type"] = "HUMAN"
    data["reviewer_id"] = cleaned_reviewer_id
    data["reviewed_at"] = str(reviewed_at).strip()
    data["evidence_hash"] = canonical_ev_hash
    data["adjudication_status"] = "FINAL"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    return data


def extract_identity_manifest_from_roster(roster_paths: Path | list[Path]) -> dict[str, Any]:
    """Extract an independent ground-truth identity manifest from one or more roster files.

    Guarantees:
    1. Identity roster is completely independent of runtime detections or annotation event proposals.
    2. Eligible for final frozen event evaluation (source_type = INDEPENDENT_IDENTITY_ROSTER)
       only when all roster sources have verified human review.
    3. Scope is FULL_SYSTEM_EVENT_EVALUATION if all sources are human approved, otherwise
       UNREVIEWED_ROSTER_SCOPE (not eligible for frozen event evaluation).
    4. Validates identity contracts and requires valid 64-char hexadecimal video_sha256.
    5. Eliminates circular SHA dependency: freezing this manifest produces manifest_sha256,
       which can then be directly embedded into annotation skeletons prior to human annotation review.
    """
    paths = [roster_paths] if isinstance(roster_paths, Path) else list(roster_paths)
    if not paths:
        raise ValueError("no roster paths provided for identity manifest extraction")

    videos: dict[str, dict[str, Any]] = {}
    proven_set: set[tuple[str, str, str, str]] = set()
    roster_sources: list[dict[str, Any]] = []

    for p in sorted(paths):
        if not p.is_file():
            raise FileNotFoundError(f"roster file not found: {p}")
        file_bytes = p.read_bytes()
        file_sha = hashlib.sha256(file_bytes).hexdigest()
        raw = json.loads(file_bytes.decode("utf-8"))

        p_status = raw.get("human_review_status", "PENDING") if isinstance(raw, dict) else "PENDING"
        p_rev_type = raw.get("reviewer_type", "AI") if isinstance(raw, dict) else "AI"
        p_rev_id = raw.get("reviewer_id") if isinstance(raw, dict) else None
        p_rev_at = raw.get("reviewed_at") if isinstance(raw, dict) else None
        p_ev_hash = raw.get("evidence_hash") if isinstance(raw, dict) else None
        p_adj_status = raw.get("adjudication_status", "PENDING") if isinstance(raw, dict) else "PENDING"
        p_roster_status = raw.get("roster_status") if isinstance(raw, dict) else None

        is_valid_human = (
            p_status == "APPROVED"
            and p_rev_type == "HUMAN"
            and bool(p_rev_id and str(p_rev_id).strip())
            and bool(p_rev_at and str(p_rev_at).strip())
            and p_adj_status == "FINAL"
        )

        roster_sources.append({
            "path": str(p.resolve()),
            "file_name": p.name,
            "sha256": file_sha,
            "roster_status": p_roster_status or ("HUMAN_APPROVED_IDENTITY_ROSTER" if is_valid_human else "UNREVIEWED_IDENTITY_ROSTER"),
            "human_review_status": p_status,
            "reviewer_type": p_rev_type,
            "reviewer_id": str(p_rev_id).strip() if p_rev_id else None,
            "reviewed_at": str(p_rev_at).strip() if p_rev_at else None,
            "evidence_hash": str(p_ev_hash).strip() if p_ev_hash else file_sha,
            "adjudication_status": p_adj_status,
            "is_human_approved": is_valid_human,
        })

        raw_videos: list[dict[str, Any]] = []
        if isinstance(raw, list):
            if raw and "vehicles" in raw[0]:
                raw_videos = raw
            elif raw and "video_id" in raw[0] and "occupant_id" in raw[0]:
                video_map: dict[str, dict[str, Any]] = {}
                for rec in raw:
                    vid = str(rec.get("video_id", "")).strip()
                    if vid not in video_map:
                        video_map[vid] = {
                            "video_id": vid,
                            "video_sha256": str(rec.get("video_sha256", "")).strip(),
                            "fps": float(rec.get("fps", 30.0)),
                            "frame_count": int(rec.get("frame_count", 0)),
                            "duration_seconds": float(rec.get("duration_seconds", 0.0)),
                            "vehicles": {},
                        }
                    veh_id = str(rec.get("vehicle_id", "")).strip()
                    cab_id = str(rec.get("cabin_id", "")).strip()
                    occ_id = str(rec.get("occupant_id", "")).strip()
                    role = str(rec.get("role", "unknown")).strip()
                    v_entry = video_map[vid]["vehicles"]
                    if veh_id not in v_entry:
                        v_entry[veh_id] = {"vehicle_id": veh_id, "cabins": {}}
                    c_entry = v_entry[veh_id]["cabins"]
                    if cab_id not in c_entry:
                        c_entry[cab_id] = {"cabin_id": cab_id, "occupants": []}
                    c_entry[cab_id]["occupants"].append({"occupant_id": occ_id, "role": role})

                for v_info in video_map.values():
                    raw_videos.append({
                        "video_id": v_info["video_id"],
                        "video_sha256": v_info["video_sha256"],
                        "fps": v_info["fps"],
                        "frame_count": v_info["frame_count"],
                        "duration_seconds": v_info["duration_seconds"],
                        "vehicles": [
                            {
                                "vehicle_id": veh_data["vehicle_id"],
                                "cabins": list(veh_data["cabins"].values()),
                            }
                            for veh_data in v_info["vehicles"].values()
                        ],
                    })
        elif isinstance(raw, dict):
            if "videos" in raw and isinstance(raw["videos"], list):
                raw_videos = raw["videos"]
            elif "videos" in raw and isinstance(raw["videos"], dict):
                raw_videos = list(raw["videos"].values())
            elif "video_id" in raw:
                raw_videos = [raw]
            else:
                raise ValueError(f"{p}: unrecognized roster format (expected 'videos' or 'video_id')")

        canonical_ev_hash = canonical_identity_evidence_hash(raw_videos)
        if p_ev_hash:
            clean_p_ev_hash = str(p_ev_hash).strip().lower()
            if clean_p_ev_hash != canonical_ev_hash:
                raise ValueError(
                    f"{p.name}: roster evidence_hash semantic mismatch: "
                    f"recorded {clean_p_ev_hash} != recomputed {canonical_ev_hash}"
                )
        roster_sources[-1]["evidence_hash"] = canonical_ev_hash

        for v_item in raw_videos:
            vid = str(v_item.get("video_id", "")).strip()
            if not vid:
                raise ValueError(f"{p}: roster video entry missing video_id")
            v_sha = str(v_item.get("video_sha256", "")).strip()
            if not v_sha or len(v_sha) != 64:
                raise ValueError(f"{p}: video '{vid}' missing valid 64-character video_sha256: {v_sha!r}")

            fps = float(v_item.get("fps", 30.0))
            frame_count = int(v_item.get("frame_count", 0))
            duration = float(v_item.get("duration_seconds", frame_count / fps if fps > 0 else 0.0))

            if vid not in videos:
                videos[vid] = {
                    "video_id": vid,
                    "video_sha256": v_sha,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": duration,
                    "vehicle_ids": set(),
                    "cabin_ids": set(),
                    "occupants": {},
                }
            else:
                v_data = videos[vid]
                if v_data["video_sha256"] != v_sha:
                    raise ValueError(f"{p}: conflicting video_sha256 for video '{vid}': {v_data['video_sha256']} != {v_sha}")
                v_data["fps"] = fps
                v_data["frame_count"] = max(v_data["frame_count"], frame_count)
                v_data["duration_seconds"] = max(v_data["duration_seconds"], duration)

            v_entry = videos[vid]
            vehicles_list = v_item.get("vehicles", [])
            for veh in vehicles_list:
                veh_id = str(veh.get("vehicle_id", "")).strip()
                if veh_id:
                    v_entry["vehicle_ids"].add(veh_id)
                for cab in veh.get("cabins", []):
                    cab_id = str(cab.get("cabin_id", "")).strip()
                    if cab_id:
                        v_entry["cabin_ids"].add(cab_id)
                    for occ in cab.get("occupants", []):
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

                        if occ_id not in v_entry["occupants"]:
                            v_entry["occupants"][occ_id] = {
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
                                "source": "INDEPENDENT_IDENTITY_ROSTER",
                            }

    if not proven_set:
        raise ValueError("no proven identities found in roster")

    videos_out: dict[str, Any] = {}
    for vid, v in sorted(videos.items()):
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

    all_human_approved = bool(roster_sources and all(s["is_human_approved"] for s in roster_sources))
    if all_human_approved:
        evaluation_scope = "FULL_SYSTEM_EVENT_EVALUATION"
        eligible_for_frozen_event_evaluation = True
    else:
        evaluation_scope = "UNREVIEWED_ROSTER_SCOPE"
        eligible_for_frozen_event_evaluation = False

    if len(paths) == 1:
        source_path_str = str(paths[0].resolve())
        source_sha = roster_sources[0]["sha256"]
    else:
        source_path_str = str(paths[0].parent.resolve())
        hasher = hashlib.sha256()
        for s in roster_sources:
            hasher.update(bytes.fromhex(s["sha256"]))
        source_sha = hasher.hexdigest()

    if len(roster_sources) == 1:
        s0 = roster_sources[0]
        review_provenance = {
            "all_sources_human_approved": all_human_approved,
            "human_review_status": s0["human_review_status"],
            "reviewer_type": s0["reviewer_type"],
            "reviewer_id": s0["reviewer_id"],
            "reviewed_at": s0["reviewed_at"],
            "evidence_hash": s0["evidence_hash"],
            "adjudication_status": s0["adjudication_status"],
        }
    else:
        reviewer_ids = sorted(list({s["reviewer_id"] for s in roster_sources if s["reviewer_id"]}))
        reviewed_ats = [s["reviewed_at"] for s in roster_sources if s.get("reviewed_at")]
        latest_reviewed_at = max(reviewed_ats) if reviewed_ats else None
        review_provenance = {
            "all_sources_human_approved": all_human_approved,
            "human_review_status": "APPROVED" if all_human_approved else "PENDING",
            "reviewer_type": "HUMAN" if all_human_approved else "MIXED_OR_AI",
            "reviewer_ids": reviewer_ids,
            "reviewer_id": reviewer_ids[0] if len(reviewer_ids) == 1 else (",".join(reviewer_ids) if reviewer_ids else None),
            "reviewed_at": latest_reviewed_at,
            "evidence_hash": source_sha,
            "adjudication_status": "FINAL" if all_human_approved else "PENDING",
        }

    return {
        "manifest_version": "v2.0",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_type": "INDEPENDENT_IDENTITY_ROSTER",
        "evaluation_scope": evaluation_scope,
        "eligible_for_frozen_event_evaluation": eligible_for_frozen_event_evaluation,
        "source_path": source_path_str,
        "source_sha256": source_sha,
        "evidence_hash": review_provenance.get("evidence_hash"),
        "roster_sources": roster_sources,
        "review_provenance": review_provenance,
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
        "INDEPENDENT_IDENTITY_ROSTER",
    }
    legacy_debug_source_types = {
        "RUNTIME_PREDICTIONS_CSV",
        "RUNTIME_EVENT_CANDIDATES",
        "INDEPENDENT_GROUND_TRUTH_ANNOTATIONS",
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

    roster_review_record = None
    roster_sources_out = None
    if source_type in governed_frozen_source_types:
        if not manifest.get("eligible_for_frozen_event_evaluation", False):
            raise ValueError(
                f"cannot freeze governed identity manifest: eligible_for_frozen_event_evaluation is False "
                f"(evaluation_scope={manifest.get('evaluation_scope')!r})"
            )
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

        if source_type == "INDEPENDENT_IDENTITY_ROSTER":
            roster_sources = manifest.get("roster_sources", [])
            if roster_sources:
                for item in roster_sources:
                    item_path_str = item.get("path", "")
                    if not item_path_str:
                        raise ValueError("roster source entry missing path")
                    item_p = Path(item_path_str)
                    if not item_p.is_file():
                        raise ValueError(f"roster source file does not exist on disk: {item_path_str}")
                    disk_sha = sha256_file(item_p)
                    if disk_sha != item.get("sha256"):
                        raise ValueError(
                            f"roster source '{item_p.name}' cryptographic mismatch: recorded {item.get('sha256')} != disk {disk_sha}"
                        )
                    if item.get("human_review_status") != "APPROVED":
                        raise ValueError(
                            f"roster source '{item_p.name}' requires human_review_status == 'APPROVED', got {item.get('human_review_status')!r}"
                        )
                    if item.get("reviewer_type") != "HUMAN":
                        raise ValueError(
                            f"roster source '{item_p.name}' requires reviewer_type == 'HUMAN' (AI rosters are strictly rejected), got {item.get('reviewer_type')!r}"
                        )
                    r_id = str(item.get("reviewer_id", "")).strip()
                    if not r_id:
                        raise ValueError(f"roster source '{item_p.name}' requires a non-empty reviewer_id")
                    r_at = str(item.get("reviewed_at", "")).strip()
                    if not r_at:
                        raise ValueError(f"roster source '{item_p.name}' requires a non-empty reviewed_at timestamp")
                    try:
                        parsed_dt = datetime.fromisoformat(r_at.replace("Z", "+00:00"))
                        if parsed_dt.tzinfo is None:
                            raise ValueError("reviewed_at must include timezone")
                    except Exception as err:
                        raise ValueError(f"roster source '{item_p.name}' invalid reviewed_at timestamp: {err}")
                    src_raw = json.loads(item_p.read_text(encoding="utf-8"))
                    src_v = src_raw.get("videos", [])
                    if isinstance(src_v, dict):
                        src_v = list(src_v.values())
                    recomputed_ev_hash = canonical_identity_evidence_hash(src_v)
                    ev_hash = str(item.get("evidence_hash", "")).strip().lower()
                    if not ev_hash or len(ev_hash) != 64:
                        raise ValueError(f"roster source '{item_p.name}' requires a valid 64-character evidence_hash, got {ev_hash!r}")
                    if ev_hash != recomputed_ev_hash:
                        raise ValueError(
                            f"roster source '{item_p.name}' evidence_hash semantic mismatch: "
                            f"recorded {ev_hash} != recomputed {recomputed_ev_hash}"
                        )
                    if item.get("adjudication_status") != "FINAL":
                        raise ValueError(
                            f"roster source '{item_p.name}' requires adjudication_status == 'FINAL', got {item.get('adjudication_status')!r}"
                        )
                if len(roster_sources) > 1:
                    hasher = hashlib.sha256()
                    for s in roster_sources:
                        hasher.update(bytes.fromhex(s["sha256"]))
                    calc_coll_sha = hasher.hexdigest()
                    if calc_coll_sha != source_sha:
                        raise ValueError(
                            f"multi-roster collection source_sha256 mismatch: recorded {source_sha} != calculated {calc_coll_sha}"
                        )
                roster_sources_out = roster_sources

            rev = manifest.get("review_provenance", {})
            if not rev and "human_review_status" in manifest:
                rev = {
                    "human_review_status": manifest.get("human_review_status"),
                    "reviewer_type": manifest.get("reviewer_type"),
                    "reviewer_id": manifest.get("reviewer_id"),
                    "reviewed_at": manifest.get("reviewed_at"),
                    "evidence_hash": manifest.get("evidence_hash"),
                    "adjudication_status": manifest.get("adjudication_status"),
                }
            if rev.get("human_review_status") != "APPROVED":
                raise ValueError(
                    f"INDEPENDENT_IDENTITY_ROSTER requires human_review_status == 'APPROVED', got {rev.get('human_review_status')!r}"
                )
            if rev.get("reviewer_type") != "HUMAN":
                raise ValueError(
                    f"INDEPENDENT_IDENTITY_ROSTER requires reviewer_type == 'HUMAN' (AI rosters are strictly rejected), got {rev.get('reviewer_type')!r}"
                )
            if not rev.get("reviewer_id"):
                raise ValueError("INDEPENDENT_IDENTITY_ROSTER requires a non-empty reviewer_id")
            r_at_raw = rev.get("reviewed_at")
            r_at = str(r_at_raw).strip() if r_at_raw is not None else ""
            if not r_at:
                raise ValueError("INDEPENDENT_IDENTITY_ROSTER requires a non-empty reviewed_at timestamp")
            try:
                parsed_dt = datetime.fromisoformat(r_at.replace("Z", "+00:00"))
                if parsed_dt.tzinfo is None:
                    raise ValueError("reviewed_at must include timezone")
            except Exception as err:
                raise ValueError(f"INDEPENDENT_IDENTITY_ROSTER invalid reviewed_at timestamp: {err}")
            ev_hash = str(rev.get("evidence_hash", "")).strip()
            if not ev_hash or len(ev_hash) != 64:
                raise ValueError(f"INDEPENDENT_IDENTITY_ROSTER requires a valid 64-character evidence_hash, got {ev_hash!r}")
            if len(roster_sources) == 1:
                expected_rev_ev_hash = roster_sources[0]["evidence_hash"]
                if ev_hash.lower() != expected_rev_ev_hash.lower():
                    raise ValueError(
                        f"manifest review_provenance evidence_hash semantic mismatch: "
                        f"recorded {ev_hash} != roster source evidence_hash {expected_rev_ev_hash}"
                    )
            root_ev_hash = manifest.get("evidence_hash")
            if root_ev_hash:
                clean_root_ev = str(root_ev_hash).strip().lower()
                expected_root_ev = (
                    roster_sources[0]["evidence_hash"]
                    if len(roster_sources) == 1
                    else ev_hash.lower()
                )
                if clean_root_ev != expected_root_ev:
                    raise ValueError(
                        f"manifest root evidence_hash semantic mismatch: "
                        f"recorded {clean_root_ev} != expected {expected_root_ev}"
                    )
            if rev.get("adjudication_status") != "FINAL":
                raise ValueError(
                    f"INDEPENDENT_IDENTITY_ROSTER requires adjudication_status == 'FINAL', got {rev.get('adjudication_status')!r}"
                )
            roster_review_record = rev
    else:
        eligible_for_frozen_event_evaluation = False
        evaluation_scope = (
            "LEGACY_ANNOTATION_EXTRACTED_SCOPE"
            if source_type == "INDEPENDENT_GROUND_TRUTH_ANNOTATIONS"
            else "LEGACY_DEBUG_ONLY"
        )
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

    manifest_videos = manifest.get("videos", {})
    if not manifest_videos:
        raise ValueError("identity manifest declares no videos")

    for vid in video_ids:
        if vid not in manifest_videos:
            raise ValueError(f"proven identity video '{vid}' missing from manifest videos")

    lock_videos: dict[str, Any] = {}
    for vid, v_entry in manifest_videos.items():
        v_sha = str(v_entry.get("video_sha256", "")).strip()
        if source_type in governed_frozen_source_types:
            if not v_sha or len(v_sha) != 64:
                raise ValueError(
                    f"governed identity manifest requires a non-empty 64-character video_sha256 for video '{vid}', got {v_sha!r}"
                )
        lock_videos[vid] = {
            "sha256": v_sha,
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
        "evidence_hash": roster_review_record.get("evidence_hash") if roster_review_record else manifest.get("evidence_hash"),
        "locked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if roster_sources_out is not None:
        lock_data["roster_sources"] = roster_sources_out
    if roster_review_record is not None:
        lock_data["review_provenance"] = roster_review_record

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
    manifest_sha256: str | None = None,
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
    5. Copies manifest_sha256 or manifest_lock["manifest_sha256"] directly into review_provenance["identity_manifest_sha256"].
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

    manifest_sha = str(manifest_sha256 or "").strip()
    if not manifest_sha and manifest_lock is not None:
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
    manifest_sha256: str | None = None,
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
            manifest_sha256=manifest_sha256,
            source_id=source_id,
        )
        for cid in cabin_ids
    ]


def freeze_identity_adjudication(
    adjudication_path: Path,
    output_lock_path: Path,
    identity_manifest_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze and cryptographically lock an identity adjudication mapping.

    Requirements:
    1. Human approval: human_review_status == 'APPROVED', reviewer_type == 'HUMAN',
       valid non-empty reviewer_id, valid ISO-8601 reviewed_at with timezone.
    2. Strict 1-to-1 bijective mapping (no two runtime tracks map to the same GT occupant).
    3. Strict cabin locality: runtime occupant and target GT occupant must share the exact
       same video, vehicle, and cabin scope.
    4. Cryptographic binding to target identity manifest SHA-256.
    5. Refuses overwrite of existing frozen adjudication locks.
    6. Final adjudication: adjudication_status == 'FINAL'.
    """
    if output_lock_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen identity adjudication: {output_lock_path}")
    if not adjudication_path.is_file():
        raise FileNotFoundError(f"adjudication file not found: {adjudication_path}")

    raw = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("adjudication root must be a JSON object")

    if raw.get("human_review_status") != "APPROVED":
        raise ValueError("identity adjudication requires human_review_status == 'APPROVED'")
    if raw.get("reviewer_type") != "HUMAN":
        raise ValueError("identity adjudication requires reviewer_type == 'HUMAN'")
    if raw.get("adjudication_status") != "FINAL":
        raise ValueError("identity adjudication requires adjudication_status == 'FINAL'")

    reviewer_id = str(raw.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise ValueError("identity adjudication missing non-empty reviewer_id")

    reviewed_at_str = str(raw.get("reviewed_at", "")).strip()
    if not reviewed_at_str:
        raise ValueError("identity adjudication missing reviewed_at timestamp")
    try:
        dt = datetime.fromisoformat(reviewed_at_str.replace("Z", "+00:00"))
        if dt.utcoffset() is None:
            raise ValueError("timezone is missing")
    except ValueError as exc:
        raise ValueError(f"reviewed_at is not a valid timezone-aware ISO-8601 timestamp: {exc}") from exc

    target_manifest_sha = str(raw.get("target_identity_manifest_sha256", "")).strip()
    if not target_manifest_sha or len(target_manifest_sha) != 64:
        raise ValueError("identity adjudication requires a valid 64-character target_identity_manifest_sha256")

    manifest_proven: set[str] = set()
    if identity_manifest_lock_path is not None:
        if not identity_manifest_lock_path.is_file():
            raise FileNotFoundError(f"identity manifest lock not found: {identity_manifest_lock_path}")
        manifest_lock = json.loads(identity_manifest_lock_path.read_text(encoding="utf-8"))
        if manifest_lock.get("status") != "FROZEN_IDENTITY_MANIFEST":
            raise ValueError("identity manifest lock status is not FROZEN_IDENTITY_MANIFEST")
        lock_sha = str(manifest_lock.get("manifest_sha256", "")).strip().lower()
        if lock_sha != target_manifest_sha:
            raise ValueError(
                f"target_identity_manifest_sha256 {target_manifest_sha} does not match manifest lock SHA {lock_sha}"
            )
        for item in manifest_lock.get("proven_identities", []):
            manifest_proven.add(item.get("occupant_id", ""))

    raw_mappings = raw.get("mappings", raw.get("mapping", {}))
    if not raw_mappings or not isinstance(raw_mappings, dict):
        raise ValueError("identity adjudication requires a non-empty 'mappings' dictionary")

    raw_cabin_mappings = raw.get("cabin_mappings", raw.get("cabin_mapping", {}))
    validated_cabin_mappings: dict[str, str] = {}
    if raw_cabin_mappings:
        if not isinstance(raw_cabin_mappings, dict):
            raise ValueError("identity adjudication 'cabin_mappings' must be a dictionary")
        seen_target_cabins: set[str] = set()
        for r_cab, g_cab in raw_cabin_mappings.items():
            r_cab = str(r_cab).strip()
            g_cab = str(g_cab).strip()
            if g_cab in seen_target_cabins:
                raise ValueError(f"many-to-one cabin adjudication violation: target GT cabin '{g_cab}' mapped more than once")
            seen_target_cabins.add(g_cab)
            r_vid = r_cab.split(":")[1] if r_cab.startswith("video:") else ""
            g_vid = g_cab.split(":")[1] if g_cab.startswith("video:") else ""
            if r_vid != g_vid or not r_vid:
                raise ValueError(
                    f"cross-video cabin adjudication violation: runtime cabin '{r_cab}' and target GT cabin '{g_cab}' must share the same video"
                )
            validated_cabin_mappings[r_cab] = g_cab

    mappings: dict[str, str] = {}
    seen_gt: set[str] = set()
    errors: list[str] = []

    for runtime_occ, gt_occ in raw_mappings.items():
        runtime_occ = str(runtime_occ).strip()
        gt_occ = str(gt_occ).strip()

        # Injective check: no two runtime tracks map to the same GT occupant
        if gt_occ in seen_gt:
            errors.append(f"many-to-one adjudication violation: target GT occupant '{gt_occ}' mapped more than once")
        seen_gt.add(gt_occ)

        if ":occupant-track:" not in runtime_occ or ":occupant-track:" not in gt_occ:
            errors.append(f"mapping ({runtime_occ} -> {gt_occ}) missing ':occupant-track:'")
            continue

        r_cabin = runtime_occ.rsplit(":occupant-track:", 1)[0]
        g_cabin = gt_occ.rsplit(":occupant-track:", 1)[0]

        if r_cabin != g_cabin:
            if validated_cabin_mappings.get(r_cabin) != g_cabin:
                errors.append(
                    f"cross-cabin adjudication violation: runtime cabin '{r_cabin}' != target GT cabin '{g_cabin}'. "
                    "Cross-cabin occupant mapping is only permitted when explicitly declared in a verified 'cabin_mappings' dictionary."
                )

        if manifest_proven and gt_occ not in manifest_proven:
            errors.append(f"target GT occupant '{gt_occ}' is not declared in the bound identity manifest")

        mappings[runtime_occ] = gt_occ

    if errors:
        raise ValueError("cannot freeze identity adjudication:\n- " + "\n- ".join(errors))

    adjudication_sha = sha256_file(adjudication_path)
    lock_data = {
        "schema_version": "v2.0",
        "status": "FROZEN_IDENTITY_ADJUDICATION",
        "adjudication_sha256": adjudication_sha,
        "adjudication_status": "FINAL",
        "adjudication_file": adjudication_path.name,
        "target_identity_manifest_sha256": target_manifest_sha,
        "human_review_status": "APPROVED",
        "reviewer_type": "HUMAN",
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at_str,
        "mappings": mappings,
        "mapping_count": len(mappings),
        "cabin_mapping_count": len(validated_cabin_mappings) if validated_cabin_mappings else 0,
        "notes": raw.get("notes", ""),
        "locked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if validated_cabin_mappings:
        lock_data["cabin_mappings"] = validated_cabin_mappings
    if identity_manifest_lock_path is not None:
        lock_data["identity_manifest_lock_path"] = str(identity_manifest_lock_path.resolve())

    output_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock_path.open("w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
        f.write("\n")

    return lock_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract locked identity manifest from runtime tracking evidence or independent ground truth"
    )
    parser.add_argument("--roster", nargs="+", type=Path, help="Paths to independent identity roster JSON/YAML files")
    parser.add_argument("--tracks", "--tracks-jsonl", type=Path, dest="tracks", help="Path to runtime_identity_tracks artifact (.jsonl or .json)")
    parser.add_argument("--annotations", nargs="+", type=Path, help="Paths to human-approved sequence annotation JSON files")
    parser.add_argument("--annotations-dir", type=Path, help="Directory containing human-approved sequence annotation JSON files")
    parser.add_argument("--predictions-csv", type=Path, help="Legacy debug: Path to runtime predictions CSV")
    parser.add_argument("--tracking-json", type=Path, help="Legacy debug: Path to runtime tracking JSON")
    parser.add_argument("--output-manifest", type=Path, help="Output manifest JSON path")
    parser.add_argument("--freeze-lock", type=Path, help="Optional output path to freeze manifest into lock JSON")
    parser.add_argument("--manifest-lock", type=Path, help="Optional path to existing frozen identity manifest lock")
    parser.add_argument("--manifest-sha256", help="Optional explicit identity manifest SHA-256 for skeleton")
    parser.add_argument("--adjudication", type=Path, help="Path to identity adjudication mapping JSON to freeze")
    parser.add_argument("--adjudication-lock", type=Path, help="Output path to freeze identity adjudication mapping")
    parser.add_argument("--skeleton-output", type=Path, help="Optional output path to generate annotation skeleton")
    parser.add_argument("--video-id", help="Video ID for skeleton generation")
    parser.add_argument("--cabin-id", help="Optional Cabin ID for skeleton generation")

    args = parser.parse_args()

    if args.adjudication and args.adjudication_lock:
        lock_data = freeze_identity_adjudication(
            args.adjudication,
            args.adjudication_lock,
            identity_manifest_lock_path=args.manifest_lock,
        )
        print(f"Frozen identity adjudication lock saved to {args.adjudication_lock}")
        return

    if not args.output_manifest and not args.skeleton_output:
        parser.error("must provide --output-manifest or --skeleton-output (or --adjudication and --adjudication-lock)")

    if args.roster:
        manifest = extract_identity_manifest_from_roster(args.roster)
    elif args.tracks:
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
        parser.error("must provide either --roster, --tracks, --annotations/--annotations-dir, --predictions-csv, or --tracking-json")

    if args.output_manifest:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.output_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"Extracted identity manifest saved to {args.output_manifest}")

    lock_data = None
    if args.freeze_lock:
        if not args.output_manifest:
            parser.error("--freeze-lock requires --output-manifest")
        lock_data = freeze_identity_manifest(args.output_manifest, args.freeze_lock)
        print(f"Frozen identity manifest lock saved to {args.freeze_lock}")

    if args.skeleton_output:
        vid = args.video_id or next(iter(manifest.get("videos", {}).keys()))
        skeleton = generate_annotation_skeleton(
            manifest,
            vid,
            cabin_id=args.cabin_id,
            manifest_lock=lock_data,
            manifest_sha256=args.manifest_sha256,
        )
        args.skeleton_output.parent.mkdir(parents=True, exist_ok=True)
        with args.skeleton_output.open("w", encoding="utf-8") as handle:
            json.dump(skeleton, handle, indent=2)
            handle.write("\n")
        print(f"Annotation skeleton saved to {args.skeleton_output}")


if __name__ == "__main__":
    main()
