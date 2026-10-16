"""Read-only artifact/data audit; protected subject trees are pruned before walking."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import psutil

from training.common import sha256_file
from training.dmd.holdout_guard import is_holdout_subject

OUT = Path("reports/ai85_plus")


def record(path: Path) -> dict:
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def walk(root: Path):
    for directory, dirs, names in os.walk(root):
        dirs[:] = [
            d
            for d in dirs
            if not is_holdout_subject(Path(directory) / d)
            and d not in {"__pycache__", "node_modules", ".git"}
        ]
        for name in names:
            path = Path(directory) / name
            if not is_holdout_subject(path):
                yield path


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "audit.json").exists():
        raise FileExistsError("use --output with a fresh directory to preserve audit history")
    models = []
    for folder in ("models", "model-training", "training", "runs", "reports"):
        for path in walk(Path(folder)):
            if path.suffix not in {".pt", ".pth", ".onnx"} and not (
                path.suffix == ".npz" and "models" in path.parts
            ):
                continue
            models.append(
                {
                    **record(path),
                    "architecture": None,
                    "training_dataset": None,
                    "class_mapping": None,
                    "input_resolution": None,
                    "epochs": None,
                    "version": None,
                    "validation_results": None,
                    "approval_status": "UNVERIFIED_INVENTORY_ONLY",
                }
            )
    trees = {}
    for name in ("dataset", "datasets"):
        root = Path(name)
        counts, total, files = Counter(), 0, 0
        if root.is_dir():
            for path in walk(root):
                counts[path.suffix.lower()] += 1
                total += path.stat().st_size
                files += 1
        trees[name] = {
            "exists": root.is_dir(),
            "files_excluding_protected_paths": files,
            "bytes_excluding_protected_paths": total,
            "extensions": dict(counts),
            "children": sorted(p.name for p in root.iterdir()) if root.exists() else [],
        }
    roots = Path("dataset/external_dmd/subjects")
    subject_names = sorted(p.name for p in roots.iterdir() if p.is_dir())
    policies = [record(p) for p in Path("dataset").glob("*.yaml")]
    policies += [record(p) for p in Path("dataset/schemas").glob("*.json")]
    preserved = []
    for folder in ("reports/ai85_upgrade", "reports/ai85_next", "models/active"):
        preserved.extend(record(p) for p in walk(Path(folder)))
    manifest_paths = [
        p
        for p in Path("dataset/manifests").glob("*")
        if p.is_file() and p.suffix in {".json", ".jsonl", ".yaml"} and not is_holdout_subject(p)
    ]
    manifests = [record(p) for p in manifest_paths]
    # File existence and declared old-path references only; no relabeling or rewriting manifests.
    refs = []
    import re

    for path in manifest_paths:
        if path.stat().st_size > 8_000_000:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        strings = set(re.findall(r'"(datasets/[^"\n]+)"', content))
        for value in sorted(strings):
            if is_holdout_subject(value):
                continue
            old = Path(value)
            replacement = Path("dataset") / Path(value).relative_to("datasets")
            refs.append(
                {
                    "manifest": path.as_posix(),
                    "reference": value,
                    "declared_exists": old.exists(),
                    "local_counterpart_exists": replacement.exists(),
                }
            )
    result = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "remote_head": subprocess.check_output(
            ["git", "ls-remote", "origin", "refs/heads/roadwatch-v4-candidate"], text=True
        ).strip(),
        "python": platform.python_version(),
        "ram_total_mb": psutil.virtual_memory().total / 1024**2,
        "ram_available_mb_at_audit": psutil.virtual_memory().available / 1024**2,
        "disk_free_gb": shutil.disk_usage(".").free / 1024**3,
        "model_count": len(models),
        "model_inventory": models,
        "data_trees": trees,
        "subject_directory_names_only": subject_names,
        "protected_contents_read": False,
        "policies_schemas": policies,
        "manifests": manifests,
        "legacy_manifest_references": refs,
        "history_preservation": preserved,
        "governance": "INDEPENDENT_AI85_NOT_ESTABLISHED",
        "mounted_external_training_runs": "No external mount configured; local runs inventoried.",
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"models": len(models), "data_trees": trees, "subjects": subject_names}, indent=2
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    run()
