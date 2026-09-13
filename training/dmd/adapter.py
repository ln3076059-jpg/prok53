"""Official ASAM OpenLABEL / VCD annotation adapter for Vicomtech DMD.

Extracts continuous temporal intervals for phone actions and non-target activities,
preserving source provenance without altering raw annotations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.common import sha256_file


SOURCE_ID = "EXTERNAL_DMD_VICOMTECH"
DATASET_ROLE = "TEMPORAL_DEVELOPMENT"
CANONICAL_ELIGIBLE = False
UNTOUCHED_HOLDOUT_ELIGIBLE = False
RAW_CAPTURE_BYTES = False

# Canonical mapping for phone actions
PHONE_ACTION_MAPPING = {
    "driver_actions/texting_right": "PHONE_USE",
    "driver_actions/texting_left": "PHONE_USE",
    "driver_actions/phonecall_right": "PHONE_USE",
    "driver_actions/phonecall_left": "PHONE_USE",
}

NON_TARGET_ACTIONS = {
    "driver_actions/safe_drive",
    "driver_actions/reach_side",
    "driver_actions/hair_and_makeup",
    "talking/talking",
    "gaze_on_road/looking_road",
    "gaze_on_road/not_looking_road",
    "hands_using_wheel/both",
    "hands_using_wheel/only_left",
    "hands_using_wheel/only_right",
    "hands_using_wheel/none",
    "driver_actions/unclassified",
}


@dataclass(frozen=True)
class DMDInterval:
    source_id: str
    source_group: str
    source_participant_id: str
    source_session_id: str
    source_action: str
    development_phone_state: str  # PHONE_USE or NON_TARGET
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    annotation_path: str
    annotation_sha256: str
    mapping_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DMDSessionAnnotation:
    source_id: str
    source_group: str
    source_participant_id: str
    source_session_id: str
    annotation_path: str
    annotation_sha256: str
    phone_intervals: list[DMDInterval]
    all_intervals: list[DMDInterval]
    action_types: list[str]


def parse_dmd_openlabel(
    annotation_path: Path,
    fps: float = 29.76,
    total_frames: int | None = None,
) -> DMDSessionAnnotation:
    """Parse an official DMD OpenLABEL annotation JSON file.

    Extracts phone usage intervals and non-target activity intervals.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    ann_sha = sha256_file(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # OpenLABEL root can be 'openlabel' or 'vcd'
    root = data.get("openlabel") or data.get("vcd")
    if not root or not isinstance(root, dict):
        raise ValueError(f"Invalid OpenLABEL/VCD root in: {annotation_path}")

    # Extract metadata if present
    metadata = root.get("metadata", {})
    sub_id = str(metadata.get("sub_id", "")).strip()

    # Parse filename components: e.g. gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json
    filename = annotation_path.name
    parts = filename.split("_")
    group = parts[0] if len(parts) > 0 else "unknown"
    participant = sub_id or (parts[1] if len(parts) > 1 else "unknown")
    session = parts[2] if len(parts) > 2 else "s2"

    actions = root.get("actions", {})
    phone_intervals: list[DMDInterval] = []
    all_intervals: list[DMDInterval] = []
    action_types: set[str] = set()

    for action_id, action_data in actions.items():
        atype = str(action_data.get("type", ""))
        if not atype:
            continue
        action_types.add(atype)
        is_phone = atype in PHONE_ACTION_MAPPING
        state = PHONE_ACTION_MAPPING.get(atype, "NON_TARGET")

        frame_intervals = action_data.get("frame_intervals", [])
        for interval in frame_intervals:
            start_f = int(interval.get("frame_start", 0))
            end_f = int(interval.get("frame_end", 0))
            if end_f < start_f:
                continue
            if total_frames is not None and start_f >= total_frames:
                continue
            if total_frames is not None and end_f >= total_frames:
                end_f = total_frames - 1

            start_s = round(start_f / fps, 4)
            end_s = round(end_f / fps, 4)
            dur_s = round(end_s - start_s, 4)

            item = DMDInterval(
                source_id=SOURCE_ID,
                source_group=group,
                source_participant_id=participant,
                source_session_id=session,
                source_action=atype,
                development_phone_state=state,
                start_frame=start_f,
                end_frame=end_f,
                start_seconds=start_s,
                end_seconds=end_s,
                duration_seconds=dur_s,
                annotation_path=str(annotation_path).replace("\\", "/"),
                annotation_sha256=ann_sha,
            )
            all_intervals.append(item)
            if is_phone:
                phone_intervals.append(item)

    # Sort intervals chronologically
    phone_intervals.sort(key=lambda x: x.start_frame)
    all_intervals.sort(key=lambda x: x.start_frame)

    return DMDSessionAnnotation(
        source_id=SOURCE_ID,
        source_group=group,
        source_participant_id=participant,
        source_session_id=session,
        annotation_path=str(annotation_path).replace("\\", "/"),
        annotation_sha256=ann_sha,
        phone_intervals=phone_intervals,
        all_intervals=all_intervals,
        action_types=sorted(action_types),
    )
