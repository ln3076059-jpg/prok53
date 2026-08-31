from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from training.common import sha256_file

REQUIRED = ["data.yaml", "manifest.jsonl", "split_metadata.json"]
SUPPORT_FILES = [
    Path("training/kaggle_runner.py"),
    Path("notebooks/kaggle_train_mc001.ipynb"),
    Path("requirements-kaggle.txt"),
]


def build(dataset: Path, config: Path, output: Path) -> Path:
    frozen = Path("datasets/manifests/test_frozen.json")
    if not frozen.exists():
        raise ValueError("freeze the test split before building the Kaggle bundle")
    missing = [name for name in REQUIRED if not (dataset / name).exists()]
    if missing:
        raise ValueError(f"dataset is incomplete: {missing}")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "mc001_bundle"
        shutil.copytree(dataset, root / "dataset")
        shutil.copy2(config, root / "config.yaml")
        shutil.copy2(frozen, root / "test_frozen.json")
        for support in SUPPORT_FILES:
            if not support.exists():
                raise ValueError(f"missing Kaggle support file: {support}")
            shutil.copy2(support, root / support.name)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip() or "UNCOMMITTED"
        (root / "repository_commit.txt").write_text(commit + "\n")
        expected = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}
        (root / "expected_hashes.json").write_text(json.dumps(expected, indent=2, sort_keys=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        archive_base = output.with_suffix("")
        archive = Path(shutil.make_archive(str(archive_base), "zip", root))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/governed/mc3_v1"))
    parser.add_argument("--config", type=Path, default=Path("experiments/MC_001/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("mc001_kaggle_train_bundle.zip"))
    args = parser.parse_args()
    archive = build(args.dataset, args.config, args.output)
    print(json.dumps({"bundle": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
