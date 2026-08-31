from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from training.common import sha256_file

EXPERIMENT_ID = "PHONE_BOOTSTRAP_001"
USAGE = "PROPOSAL_MODEL_ONLY"
REQUIRED_MEMBERS = {"weights/best.pt", "run_metadata.json", "test_metrics.json"}
EXPECTED_SPLITS = {
    "train": {"images": 7513, "phone_boxes": 2109},
    "val": {"images": 1106, "phone_boxes": 220},
    "test": {"images": 1109, "phone_boxes": 219},
}


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {info.filename}")
        if normalized in members:
            raise ValueError(f"duplicate archive member: {normalized}")
        members[normalized] = info
    missing = REQUIRED_MEMBERS - members.keys()
    if missing:
        raise ValueError(f"recovery archive is incomplete: {sorted(missing)}")
    return members


def _read_json(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> dict:
    if member.file_size > 10 * 1024 * 1024:
        raise ValueError(f"metadata is unexpectedly large: {member.filename}")
    value = json.loads(archive.read(member).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {member.filename}")
    return value


def validate_metadata(metadata: dict, metrics: dict) -> None:
    if metadata.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("recovery experiment_id mismatch")
    if metadata.get("usage") != USAGE:
        raise ValueError("recovery weight is not marked proposal-model-only")
    digest = metadata.get("best_weights_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("missing or invalid best_weights_sha256")
    preflight = metadata.get("preflight", {})
    if preflight.get("manifest_samples") != 9728 or preflight.get("splits") != EXPECTED_SPLITS:
        raise ValueError("recovery preflight does not match the frozen phone bootstrap dataset")
    if metadata.get("test_metrics") != metrics:
        raise ValueError("test_metrics.json differs from run_metadata.json")
    if not metrics or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError("test metrics must be a non-empty object of finite numbers")


def install_recovery(archive_path: Path, destination: Path) -> dict:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    recovery_sha256 = sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        metadata = _read_json(archive, members["run_metadata.json"])
        metrics = _read_json(archive, members["test_metrics.json"])
        validate_metadata(metadata, metrics)
        if members["weights/best.pt"].file_size <= 0 or members["weights/best.pt"].file_size > 2 * 1024**3:
            raise ValueError("best.pt has an invalid size")
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "best.pt"
            with archive.open(members["weights/best.pt"]) as source, candidate.open("wb") as target:
                shutil.copyfileobj(source, target)
            weight_sha256 = sha256_file(candidate)
            if weight_sha256 != metadata["best_weights_sha256"]:
                raise ValueError("best.pt checksum differs from run metadata")

            destination = destination.resolve()
            destination.mkdir(parents=True, exist_ok=True)
            installed_weight = destination / "best.pt"
            if installed_weight.exists() and sha256_file(installed_weight) != weight_sha256:
                raise FileExistsError(f"refusing to overwrite different immutable weight: {installed_weight}")
            if not installed_weight.exists():
                shutil.copy2(candidate, installed_weight)

    import_record = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "usage": USAGE,
        "installed_at": datetime.now(UTC).isoformat(),
        "source_archive": str(archive_path),
        "recovery_sha256": recovery_sha256,
        "best_weights_sha256": metadata["best_weights_sha256"],
        "best_weights_path": str(installed_weight),
        "test_metrics": metrics,
        "preflight": metadata["preflight"],
        "policy": "Proposal generation only. Human review, vehicle context, and occupant role remain mandatory.",
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "import_record.json").write_text(
        json.dumps(import_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "best.pt.sha256").write_text(metadata["best_weights_sha256"] + "\n", encoding="utf-8")
    return import_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and install PHONE_BOOTSTRAP_001 Kaggle recovery")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--destination", type=Path, default=Path("models/proposals/phone_bootstrap_001")
    )
    args = parser.parse_args()
    print(json.dumps(install_recovery(args.archive, args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
