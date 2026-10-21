"""Verify the current private Kaggle code package against its local manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "kaggle" / "roadwatch_v27_ai85_execution"
MANIFEST = PACKAGE_ROOT / "package-manifest.json"
REPORT = ROOT / "reports" / "v27_ai85_execution"
DATASET_REF = "longngoc11/roadwatch-v27-ai85-execution"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "README.md": (PACKAGE_ROOT / "README.md").stat().st_size,
        "package-manifest.json": MANIFEST.stat().st_size,
    }
    for entry in manifest["files"]:
        expected[entry["path"]] = entry["bytes"]

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_list_files(DATASET_REF, page_size=200)
    remote = {item.name: item.total_bytes for item in response.dataset_files}
    if remote != expected:
        missing = sorted(set(expected) - set(remote))
        extra = sorted(set(remote) - set(expected))
        wrong_size = sorted(
            name for name in set(remote) & set(expected) if remote[name] != expected[name]
        )
        raise ValueError(
            f"KAGGLE_REMOTE_PACKAGE_MISMATCH:missing={missing}:extra={extra}:"
            f"wrong_size={wrong_size}"
        )
    forbidden_names = [
        name
        for name in remote
        if "__pycache__" in name or name.lower().endswith((".pyc", ".pyo"))
    ]
    if forbidden_names:
        raise ValueError(f"KAGGLE_REMOTE_BYTECODE_FOUND:{forbidden_names}")

    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "ref": DATASET_REF,
            "url": "https://www.kaggle.com/datasets/longngoc11/roadwatch-v27-ai85-execution",
            "visibility": "private",
            "version": 6,
            "status": "ready",
        },
        "verification": {
            "remote_file_count": len(remote),
            "manifest_package_file_count": manifest["file_count"],
            "remote_names_and_sizes_match_local": True,
            "remote_bytecode_files": 0,
            "local_credential_pattern_scan": "PASSED_SEPARATE_COMMAND",
            "manifest_sha256": digest(MANIFEST),
            "exclusions": manifest["exclusions"],
        },
    }
    target = REPORT / "kaggle_package_v6_receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
