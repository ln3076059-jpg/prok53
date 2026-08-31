from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from training.common import sha256_file

REQUIRED_DATASET = ["data.yaml", "manifest.jsonl", "BOOTSTRAP_MANIFEST.json"]
SUPPORT_FILES = [
    Path("training/mc_bootstrap_kaggle_runner.py"),
    Path("notebooks/kaggle_train_mc_bootstrap.ipynb"),
    Path("requirements-kaggle.txt"),
]


def build(dataset: Path, audit: Path, config: Path, output: Path) -> Path:
    missing = [name for name in REQUIRED_DATASET if not (dataset / name).is_file()]
    if missing:
        raise ValueError(f"MC bootstrap dataset is incomplete: {missing}")
    audit_report = json.loads(audit.read_text(encoding="utf-8"))
    if audit_report.get("status") != "PASS":
        raise ValueError("MC bootstrap audit must pass before bundling")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "mc_bootstrap_bundle"
        shutil.copytree(dataset, root / "dataset")
        shutil.copy2(audit, root / "audit_report.json")
        shutil.copy2(config, root / "config.yaml")
        for support in SUPPORT_FILES:
            if not support.is_file():
                raise ValueError(f"missing support file: {support}")
            shutil.copy2(support, root / support.name)
        expected = {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        (root / "MC_BOOTSTRAP_EXPECTED_HASHES.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        archive = Path(shutil.make_archive(str(output.with_suffix("")), "zip", root))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private Kaggle three-class bootstrap bundle")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/derived/mc_bootstrap_v1"))
    parser.add_argument("--audit", type=Path, default=Path("reports/mc_bootstrap_v1/data_quality.json"))
    parser.add_argument("--config", type=Path, default=Path("experiments/MC_BOOTSTRAP_001/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("kaggle/mc_bootstrap_001_kaggle_bundle.zip"))
    args = parser.parse_args()
    archive = build(args.dataset, args.audit, args.config, args.output)
    print(json.dumps({"bundle": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
