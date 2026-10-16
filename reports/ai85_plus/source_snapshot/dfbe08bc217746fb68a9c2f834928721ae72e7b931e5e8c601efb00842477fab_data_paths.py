"""Resolve repository data without moving it or silently replacing explicit paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_path(relative: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Select an explicit data root, or the single existing conventional root."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("data path must be relative and remain within its root")
    configured = os.environ.get("ROADWATCH_DATA_ROOT")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = project_root / root
        if not root.is_dir():
            raise FileNotFoundError(f"configured ROADWATCH_DATA_ROOT does not exist: {root}")
    else:
        roots = [project_root / name for name in ("dataset", "datasets")]
        existing = [root for root in roots if root.is_dir()]
        if len(existing) != 1:
            raise ValueError("set ROADWATCH_DATA_ROOT: expected exactly one dataset(s) directory")
        root = existing[0]
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("data path escapes root through a link")
    return resolved
