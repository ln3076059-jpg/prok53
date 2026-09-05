from __future__ import annotations

import re

VEHICLE_ID_PATTERN = re.compile(r"^video:[^:]+:(vehicle-track:[0-9]+|provided-vehicle)$")
CABIN_ID_PATTERN = re.compile(r"^video:[^:]+:(vehicle-track:[0-9]+:cabin:[0-9]+|provided-cabin)$")
OCCUPANT_ID_PATTERN = re.compile(
    r"^video:[^:]+:(vehicle-track:[0-9]+:cabin:[0-9]+|provided-cabin):occupant-track:[0-9]+$"
)


def validate_identity_contract(
    video_id: str,
    vehicle_id: str,
    cabin_id: str,
    occupant_id: str | None = None,
) -> list[str]:
    """Validate that video_id, vehicle_id, cabin_id, and occupant_id adhere to deterministic
    namespace patterns and hierarchical cross-field consistency.
    """
    errors: list[str] = []

    if not VEHICLE_ID_PATTERN.match(vehicle_id):
        errors.append(f"vehicle_id '{vehicle_id}' does not match pattern '{VEHICLE_ID_PATTERN.pattern}'")
    if not CABIN_ID_PATTERN.match(cabin_id):
        errors.append(f"cabin_id '{cabin_id}' does not match pattern '{CABIN_ID_PATTERN.pattern}'")

    expected_prefix = f"video:{video_id}:"
    if not vehicle_id.startswith(expected_prefix):
        errors.append(
            f"vehicle_id '{vehicle_id}' does not match expected prefix '{expected_prefix}'"
        )

    if ":vehicle-track:" in vehicle_id:
        if not cabin_id.startswith(vehicle_id + ":cabin:"):
            errors.append(
                f"cabin_id '{cabin_id}' does not belong to vehicle_id '{vehicle_id}'"
            )
    else:
        if vehicle_id != f"{expected_prefix}provided-vehicle":
            errors.append(
                f"vehicle_id '{vehicle_id}' must be '{expected_prefix}provided-vehicle'"
            )
        if cabin_id != f"{expected_prefix}provided-cabin":
            errors.append(
                f"cabin_id '{cabin_id}' must be '{expected_prefix}provided-cabin'"
            )

    if occupant_id is not None:
        if not OCCUPANT_ID_PATTERN.match(occupant_id):
            errors.append(f"occupant_id '{occupant_id}' does not match pattern '{OCCUPANT_ID_PATTERN.pattern}'")
        expected_occ_prefix = f"{cabin_id}:occupant-track:"
        if not occupant_id.startswith(expected_occ_prefix):
            errors.append(
                f"occupant_id '{occupant_id}' does not belong to cabin_id '{cabin_id}'"
            )

    return errors
