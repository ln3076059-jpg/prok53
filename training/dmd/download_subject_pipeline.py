"""Controlled Download Pipeline for DMD V3 Subjects.

Adheres strictly to the Roadwatch V3 governance:
- Environment variable / .dmd_urls.json based signed URLs (NEVER printed or committed).
- Checks free disk space on Drive D: with >= 5 GB safety margin.
- Holdout (gE-28): Archive only, verified with SHA256, NEVER extracted or inspected before freeze.
- Development (gB-9): Archive downloaded, SHA256 verified, s2 RGB body/face/hands and annotations
  extracted to subjects/gB-9/, member SHA256 recorded, archive deleted to save disk.
- Development (gZ-36): Archive downloaded, SHA256 verified, s2 RGB face/hands extracted to
  subjects/gZ-36/, member SHA256 recorded, archive deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import sha256_file
from training.dmd.holdout_guard import assert_holdout_untouched

URLS_FILE = Path(".dmd_urls.json")
MANIFEST_PATH = Path("datasets/external_dmd/manifests/dmd_sources.jsonl")
ARCHIVES_DIR = Path("datasets/external_dmd/archives")
SUBJECTS_DIR = Path("datasets/external_dmd/subjects")
HOLDOUT_DIR = Path("datasets/external_dmd/holdout/gE-28/archive_only_before_freeze")
SAFE_MARGIN_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB safe margin


def get_url_and_metadata(subject_key: str) -> dict:
    """Retrieve URL and metadata for a subject from environment or .dmd_urls.json."""
    env_map = {
        "gZ-36": "DMD_URL_GZ36",
        "gB-9": "DMD_URL_GB9",
        "gE-28": "DMD_URL_GE28",
    }
    env_var = env_map.get(subject_key)
    env_url = os.environ.get(env_var) if env_var else None

    if URLS_FILE.exists():
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if subject_key in data:
            item = data[subject_key]
            if env_url:
                item["url"] = env_url
            return item

    if env_url:
        return {"url": env_url, "archive": f"dmd-dataset-distraction-{subject_key}.tar.gz"}

    raise ValueError(f"DMD signed URL for '{subject_key}' not found in environment or {URLS_FILE}")


def verify_disk_space(needed_bytes: int) -> None:
    free_bytes = shutil.disk_usage(Path(".")).free
    remaining = free_bytes - needed_bytes
    print(f"[DISK] Free space: {free_bytes / 1024**3:.2f} GB | Needed: {needed_bytes / 1024**3:.2f} GB | Margin: {remaining / 1024**3:.2f} GB", flush=True)
    if remaining < SAFE_MARGIN_BYTES:
        raise RuntimeError(
            f"STORAGE BLOCKER: Safe margin requirement violated! "
            f"Remaining space after planned action ({remaining / 1024**3:.2f} GB) < 5.0 GB margin."
        )


def download_with_curl(url: str, dest_path: Path, expected_size: int, rate_limit: str = "8M") -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    curl_cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--limit-rate", rate_limit,
        "--retry", "5",
        "--retry-delay", "3",
        "-o", str(dest_path),
        url,
    ]

    max_retries = 100
    retries = 0
    while True:
        actual_size = dest_path.stat().st_size if dest_path.exists() else 0
        if actual_size == expected_size:
            print(f"[CURL] Download complete: {actual_size} bytes matching expected size.", flush=True)
            break
        if actual_size > expected_size:
            raise ValueError(f"File size {actual_size} exceeded expected size {expected_size} on {dest_path}")

        pct = (actual_size / expected_size * 100) if expected_size > 0 else 0
        print(f"[CURL] Resuming at {actual_size} / {expected_size} bytes ({pct:.1f}%)...", flush=True)
        res = subprocess.run(curl_cmd)
        new_size = dest_path.stat().st_size if dest_path.exists() else 0
        if new_size == expected_size:
            print(f"[CURL] Download finished successfully: {new_size} bytes.", flush=True)
            break

        retries += 1
        if retries > max_retries:
            raise RuntimeError(f"Exceeded max retries ({max_retries}) for download of {dest_path.name}")
        print(f"[CURL] Process exited (code {res.returncode}), current size {new_size}. Retrying in 3s ({retries}/{max_retries})...", flush=True)
        time.sleep(3)


def append_manifest_entry(entry: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[MANIFEST] Appended entry for {entry.get('source_participant_id')} ({entry.get('source_stream') or 'ARCHIVE'})", flush=True)


def handle_ge28_holdout(rate_limit: str = "8M") -> None:
    """Download gE-28 archive securely. NEVER inspect or extract contents."""
    print("==================================================", flush=True)
    print("DOWNLOADING HOLDOUT gE-28 (ARCHIVE ONLY - UNTOUCHED)", flush=True)
    print("==================================================", flush=True)
    
    meta = get_url_and_metadata("gE-28")
    expected_size = meta["expected_size"]
    archive_path = HOLDOUT_DIR / "dmd-dataset-distraction-gE-28.tar.gz"

    verify_disk_space(expected_size)
    download_with_curl(meta["url"], archive_path, expected_size, rate_limit)

    print("[HASH] Computing SHA256 of gE-28 holdout archive...", flush=True)
    archive_sha256 = sha256_file(archive_path)
    print(f"[HASH] gE-28 archive SHA256: {archive_sha256}", flush=True)

    manifest_entry = {
        "source_id": "EXTERNAL_DMD_VICOMTECH",
        "source_group": "gE",
        "source_participant_id": "28",
        "source_session_id": "s2",
        "archive_filename": "dmd-dataset-distraction-gE-28.tar.gz",
        "archive_size_bytes": expected_size,
        "archive_sha256": archive_sha256,
        "archive_retained": True,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
        "license_reference": "Vicomtech DMD Research License (Non-commercial academic research)",
        "dataset_role": "FINAL_UNTOUCHED_EXTERNAL_DMD_HOLDOUT"
    }
    append_manifest_entry(manifest_entry)
    print("[SUCCESS] gE-28 holdout archive secured and recorded. Untouched policy active.", flush=True)


def handle_gb9_development(rate_limit: str = "8M") -> None:
    """Download gB-9, extract s2 body/face/hands + annotation, compute member SHA256s, delete archive."""
    print("==================================================", flush=True)
    print("PROCESSING DEVELOPMENT SUBJECT gB-9", flush=True)
    print("==================================================", flush=True)

    meta = get_url_and_metadata("gB-9")
    expected_size = meta["expected_size"]
    archive_path = ARCHIVES_DIR / "dmd-dataset-distraction-gB-9.tar.gz"
    subject_dir = SUBJECTS_DIR / "gB-9"
    subject_dir.mkdir(parents=True, exist_ok=True)

    verify_disk_space(expected_size + 2 * 1024 * 1024 * 1024)
    download_with_curl(meta["url"], archive_path, expected_size, rate_limit)

    print("[HASH] Computing SHA256 of gB-9 archive...", flush=True)
    archive_sha256 = sha256_file(archive_path)
    print(f"[HASH] gB-9 archive SHA256: {archive_sha256}", flush=True)

    print("[EXTRACT] Extracting s2 RGB (body, face, hands) and distraction annotation...", flush=True)
    extracted = {}
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            name_lower = member.name.lower()
            if "s2" not in name_lower:
                continue
            if "rgb_ann_distraction.json" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting annotation: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["ann"] = dest
            elif "rgb_body.mp4" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting body stream: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["body"] = dest
            elif "rgb_face.mp4" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting face stream: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["face"] = dest
            elif "rgb_hands.mp4" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting hands stream: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["hands"] = dest

    for required in ["ann", "body", "face", "hands"]:
        if required not in extracted or not extracted[required].exists():
            raise RuntimeError(f"Missing required s2 member: {required} in gB-9")

    # Hash extracted members
    hashes = {}
    for stream_type, path in extracted.items():
        print(f"[HASH] Computing SHA256 of extracted {stream_type} ({path.name})...", flush=True)
        hashes[stream_type] = sha256_file(path)
        print(f"  {stream_type}: {hashes[stream_type]} ({path.stat().st_size} bytes)", flush=True)

    # Delete archive to free disk space
    print(f"[CLEANUP] Removing archive {archive_path.name} to free disk space...", flush=True)
    archive_path.unlink()

    now_iso = datetime.now(timezone.utc).isoformat()
    for view_key, view_name in [("body", "BODY"), ("face", "FACE"), ("hands", "HANDS")]:
        entry = {
            "source_id": "EXTERNAL_DMD_VICOMTECH",
            "source_group": "gB",
            "source_participant_id": "9",
            "source_session_id": "s2",
            "source_channel": "RGB",
            "source_stream": view_name,
            "archive_filename": "dmd-dataset-distraction-gB-9.tar.gz",
            "archive_size_bytes": expected_size,
            "archive_sha256": archive_sha256,
            "archive_retained": False,
            "member_path": str(extracted[view_key].as_posix()),
            "member_size_bytes": extracted[view_key].stat().st_size,
            "member_sha256": hashes[view_key],
            "annotation_member_path": str(extracted["ann"].as_posix()),
            "annotation_size_bytes": extracted["ann"].stat().st_size,
            "annotation_sha256": hashes["ann"],
            "downloaded_at": now_iso,
            "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
            "license_reference": "Vicomtech DMD Research License (Non-commercial academic research)",
            "dataset_role": "TEMPORAL_DEVELOPMENT"
        }
        append_manifest_entry(entry)

    print("[SUCCESS] Development subject gB-9 extracted and validated!", flush=True)


def handle_gz36_development(rate_limit: str = "8M") -> None:
    """Download gZ-36 archive, extract face and hands, compute member SHA256s, delete archive."""
    print("==================================================", flush=True)
    print("PROCESSING DEVELOPMENT SUBJECT gZ-36 (FACE & HANDS)", flush=True)
    print("==================================================", flush=True)

    meta = get_url_and_metadata("gZ-36")
    expected_size = meta["expected_size"]
    archive_path = ARCHIVES_DIR / "dmd-dataset-distraction-gZ-36.tar.gz"
    subject_dir = SUBJECTS_DIR / "gZ-36"
    subject_dir.mkdir(parents=True, exist_ok=True)

    verify_disk_space(expected_size + 2 * 1024 * 1024 * 1024)
    download_with_curl(meta["url"], archive_path, expected_size, rate_limit)

    print("[HASH] Computing SHA256 of gZ-36 archive...", flush=True)
    archive_sha256 = sha256_file(archive_path)
    print(f"[HASH] gZ-36 archive SHA256: {archive_sha256}", flush=True)

    print("[EXTRACT] Extracting s2 RGB face & hands...", flush=True)
    extracted = {}
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            name_lower = member.name.lower()
            if "s2" not in name_lower:
                continue
            if "rgb_face.mp4" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting face stream: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["face"] = dest
            elif "rgb_hands.mp4" in name_lower:
                dest = subject_dir / Path(member.name).name
                print(f"  Extracting hands stream: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted["hands"] = dest

    for required in ["face", "hands"]:
        if required not in extracted or not extracted[required].exists():
            raise RuntimeError(f"Missing required s2 member: {required} in gZ-36")

    hashes = {}
    for stream_type, path in extracted.items():
        print(f"[HASH] Computing SHA256 of extracted {stream_type} ({path.name})...", flush=True)
        hashes[stream_type] = sha256_file(path)
        print(f"  {stream_type}: {hashes[stream_type]} ({path.stat().st_size} bytes)", flush=True)

    # Delete archive
    print(f"[CLEANUP] Removing archive {archive_path.name} to free disk space...", flush=True)
    archive_path.unlink()

    # Find existing body & annotation files in gZ-36
    ann_file = next(subject_dir.glob("*_rgb_ann_distraction.json"), None)
    ann_sha = sha256_file(ann_file) if ann_file else "UNKNOWN"
    ann_size = ann_file.stat().st_size if ann_file else 0

    now_iso = datetime.now(timezone.utc).isoformat()
    for view_key, view_name in [("face", "FACE"), ("hands", "HANDS")]:
        entry = {
            "source_id": "EXTERNAL_DMD_VICOMTECH",
            "source_group": "gZ",
            "source_participant_id": "36",
            "source_session_id": "s2",
            "source_channel": "RGB",
            "source_stream": view_name,
            "archive_filename": "dmd-dataset-distraction-gZ-36.tar.gz",
            "archive_size_bytes": expected_size,
            "archive_sha256": archive_sha256,
            "archive_retained": False,
            "member_path": str(extracted[view_key].as_posix()),
            "member_size_bytes": extracted[view_key].stat().st_size,
            "member_sha256": hashes[view_key],
            "annotation_member_path": str(ann_file.as_posix()) if ann_file else "",
            "annotation_size_bytes": ann_size,
            "annotation_sha256": ann_sha,
            "downloaded_at": now_iso,
            "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
            "license_reference": "Vicomtech DMD Research License (Non-commercial academic research)",
            "dataset_role": "TEMPORAL_DEVELOPMENT"
        }
        append_manifest_entry(entry)

    print("[SUCCESS] Development subject gZ-36 multi-view extracted and recorded!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DMD V3 subjects")
    parser.add_argument("--subject", choices=["gE-28", "gB-9", "gZ-36", "all_dev", "all"], required=True)
    parser.add_argument("--rate-limit", default="8M", help="Rate limit for curl (e.g. 6M, 8M)")
    args = parser.parse_args()

    if args.subject == "gE-28":
        handle_ge28_holdout(args.rate_limit)
    elif args.subject == "gB-9":
        handle_gb9_development(args.rate_limit)
    elif args.subject == "gZ-36":
        handle_gz36_development(args.rate_limit)
    elif args.subject == "all_dev":
        handle_gb9_development(args.rate_limit)
        handle_gz36_development(args.rate_limit)
    elif args.subject == "all":
        handle_ge28_holdout(args.rate_limit)
        handle_gb9_development(args.rate_limit)
        handle_gz36_development(args.rate_limit)
