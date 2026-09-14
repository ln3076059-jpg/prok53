"""Holdout Access Guard for Roadwatch DMD Research.

Enforces strict subject isolation. Subjects designated as HOLDOUT (e.g. gE-28)
must never be opened, extracted, inspected, or inferred before explicit
pre-holdout freeze authorization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

HOLDOUT_SUBJECT_IDS = frozenset({"ge-28", "ge_28", "ge28", "28"})
FREEZE_FILE_PATH = Path("reports/DMD_V3_PRE_HOLDOUT_FREEZE.json")
AUTHORIZED_FLAG = "AUTHORIZED_FOR_ONE_SHOT_EVALUATION"


class HoldoutAccessError(RuntimeError):
    """Raised when holdout subject media or annotations are accessed before pre-holdout freeze."""
    pass


def is_holdout_subject(subject_or_path: Union[str, Path]) -> bool:
    """Check if the provided subject identifier or path corresponds to the holdout."""
    s = str(subject_or_path).lower().replace("\\", "/").strip()
    # Check direct name matches
    base = Path(s).stem.lower()
    for pattern in ["ge-28", "ge_28", "ge28"]:
        if pattern in s or pattern in base:
            return True
    return False


def is_holdout_unlocked(freeze_path: Path | None = None) -> bool:
    """Return True if pre-holdout freeze has been formally recorded and authorized."""
    target = freeze_path if freeze_path is not None else FREEZE_FILE_PATH
    if not target.exists():
        return False
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("GE28_HOLDOUT_UNLOCK") == AUTHORIZED_FLAG
    except Exception:
        return False


def assert_holdout_untouched(subject_or_path: Union[str, Path], caller_action: str = "access") -> None:
    """Enforce holdout integrity.

    Raises HoldoutAccessError if subject_or_path is holdout and freeze has not authorized access.
    """
    if is_holdout_subject(subject_or_path):
        if not is_holdout_unlocked():
            raise HoldoutAccessError(
                f"HOLDOUT ISOLATION BREACH PREVENTED: Access to holdout subject '{subject_or_path}' "
                f"is strictly prohibited before pre-holdout freeze artifact '{FREEZE_FILE_PATH}' "
                f"authorizes '{AUTHORIZED_FLAG}'. Attempted action: '{caller_action}'."
            )
