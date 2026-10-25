"""Live, fail-closed permission checks for DMD research entry points."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "dataset/external_dmd/README.md"
DEVELOPMENT = frozenset({"gB-9", "gC-14", "gZ-36"})
FIT_OPERATIONS = frozenset({"linear_head", "calibration", "fusion", "boundary", "backbone"})
LOCAL_DMD_TRAINING_OPERATION = "local_dmd_training"


def require_subject(subject: str) -> None:
    if subject not in DEVELOPMENT:
        raise PermissionError(f"Subject not eligible for development: {subject}")


def read_permissions(record: Path = RECORD) -> dict[str, str]:
    """Read the authoritative record on each call, rejecting conflicting declarations."""
    entries: dict[str, str] = {}
    for key, value in re.findall(
        r"^([A-Z][A-Z0-9_]+)\s*=\s*([^\r\n]+)$", record.read_text(encoding="utf-8"), re.M
    ):
        value = value.strip()
        if key in entries and entries[key] != value:
            raise PermissionError(f"Conflicting permission declaration: {key}")
        entries[key] = value
    return entries


def require_operation(operation: str, record: Path = RECORD) -> None:
    permissions = read_permissions(record)
    if operation == "diagnostics":
        if permissions.get("TEMPORAL_DEVELOPMENT_ELIGIBLE") != "true":
            raise PermissionError("Local temporal diagnostics are not authorized")
        return
    if operation == LOCAL_DMD_TRAINING_OPERATION:
        if permissions.get("RIGHTS_ACCEPTED") != "true":
            raise PermissionError("local_dmd_training: RIGHTS_ACCEPTED is not true")
        if permissions.get("DMD_MEDIA_LOCAL_TRAINING_ALLOWED") != "true":
            raise PermissionError(
                "local_dmd_training: explicit DMD_MEDIA_LOCAL_TRAINING_ALLOWED=true is required"
            )
        expected_scope = "DEVELOPMENT_SUBJECTS_ONLY_GB9_GC14_GZ36"
        if permissions.get("DMD_MEDIA_LOCAL_TRAINING_SCOPE") != expected_scope:
            raise PermissionError(
                "local_dmd_training: local training scope is missing or broader than expected"
            )
        return
    if operation not in FIT_OPERATIONS | {"upload"}:
        raise ValueError(f"Unsupported research operation: {operation}")
    if permissions.get("RIGHTS_ACCEPTED") != "true":
        raise PermissionError(f"{operation}: RIGHTS_ACCEPTED is not true; no fitting/upload")
    required = "EXTERNAL_UPLOAD_ALLOWED" if operation == "upload" else "TRAINING_ALLOWED"
    if permissions.get(required) != "true":
        raise PermissionError(f"{operation}: explicit {required}=true is required")
    if operation in FIT_OPERATIONS:
        for flag in ("LABELS_APPROVED", "GROUP_SPLIT_VERIFIED"):
            if permissions.get(flag) != "true":
                raise PermissionError(f"{operation}: {flag}=true is required")
