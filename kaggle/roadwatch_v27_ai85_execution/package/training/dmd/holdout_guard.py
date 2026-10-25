"""Holdout Access Guard for Roadwatch DMD Research.

Enforces strict subject isolation.
Subjects designated as CONSUMED HOLDOUTS (gZ-37 for Benchmark 002, gE-28 for Benchmark 003)
are permanently forbidden for training, threshold tuning, feature engineering,
hard-negative mining, ablation selection, calibration, or development review.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

# Consumed holdouts that are permanently forbidden for any development/tuning
CONSUMED_HOLDOUT_PATTERNS = frozenset({"gz-37", "gz_37", "gz37", "ge-28", "ge_28", "ge28"})
HOLDOUT_SUBJECT_IDS = frozenset({"ge-28", "ge_28", "ge28", "28", "gz-37", "gz_37", "gz37", "37"})

FREEZE_FILE_PATH = Path("reports/DMD_V3_PRE_HOLDOUT_FREEZE.json")
AUTHORIZED_FLAG = "AUTHORIZED_FOR_ONE_SHOT_EVALUATION"

DEVELOPMENT_FORBIDDEN_ACTIONS = frozenset({
    "train",
    "training",
    "tune",
    "tuning",
    "calibrate",
    "calibration",
    "ablation",
    "hard_negative_mining",
    "development_eval",
    "dev_eval",
    "feature_engineering",
    "development",
    "dev",
})


class HoldoutAccessError(RuntimeError):
    """Raised when holdout subject media or annotations are accessed improperly."""
    pass


def is_holdout_subject(subject_or_path: Union[str, Path]) -> bool:
    """Check if the provided subject identifier or path corresponds to any holdout subject."""
    s = str(subject_or_path).lower().replace("\\", "/").strip()
    base = Path(s).stem.lower()
    for pattern in ["ge-28", "ge_28", "ge28", "gz-37", "gz_37", "gz37"]:
        if pattern in s or pattern in base:
            return True
    return False


def is_consumed_holdout(subject_or_path: Union[str, Path]) -> bool:
    """Check if the subject is a permanently consumed holdout (gZ-37 or gE-28)."""
    return is_holdout_subject(subject_or_path)


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


def assert_not_consumed_holdout_for_development(
    subject_or_path: Union[str, Path],
    caller_action: str = "development"
) -> None:
    """Strictly block consumed holdout subjects (gZ-37, gE-28) from any development usage."""
    if is_consumed_holdout(subject_or_path):
        raise HoldoutAccessError(
            f"CONSUMED HOLDOUT BREACH PREVENTED: Subject '{subject_or_path}' is a permanently "
            f"consumed holdout (Benchmark 002/003) and is strictly forbidden for {caller_action}. "
            f"Use development pool subjects (gC-14, gZ-36, gB-9) only."
        )


def assert_holdout_untouched(
    subject_or_path: Union[str, Path],
    caller_action: str = "access"
) -> None:
    """Enforce holdout integrity.

    - Blocks development actions on consumed holdouts permanently.
    - Blocks gE-28 access before formal freeze authorization.
    """
    action_lower = caller_action.lower().replace("-", "_")
    for forbidden in DEVELOPMENT_FORBIDDEN_ACTIONS:
        if forbidden in action_lower:
            assert_not_consumed_holdout_for_development(subject_or_path, caller_action)

    if is_holdout_subject(subject_or_path):
        s = str(subject_or_path).lower().replace("\\", "/").strip()
        # gZ-37 is already consumed in Benchmark 002
        if any(p in s for p in ["gz-37", "gz_37", "gz37"]):
            if "benchmark_002" not in action_lower:
                raise HoldoutAccessError(
                    f"CONSUMED HOLDOUT BREACH: 'gZ-37' is permanently consumed by Benchmark 002 "
                    f"and cannot be re-evaluated or accessed for '{caller_action}'."
                )

        # gE-28 requires pre-holdout freeze
        if any(p in s for p in ["ge-28", "ge_28", "ge28"]):
            if not is_holdout_unlocked():
                raise HoldoutAccessError(
                    f"HOLDOUT ISOLATION BREACH PREVENTED: Access to holdout subject '{subject_or_path}' "
                    f"is strictly prohibited before pre-holdout freeze artifact '{FREEZE_FILE_PATH}' "
                    f"authorizes '{AUTHORIZED_FLAG}'. Attempted action: '{caller_action}'."
                )
