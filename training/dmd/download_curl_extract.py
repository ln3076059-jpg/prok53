"""Download gZ-37 using rate-limited curl and extract target members cleanly."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from training.common import sha256_file

SOURCE_ID = "EXTERNAL_DMD_VICOMTECH"
DATASET_ROLE = "FINAL_UNTOUCHED_DEVELOPMENT_HOLDOUT"
LICENSE_REF = "Vicomtech DMD Research License (Non-commercial academic research)"


def main():
    with open(".dmd_urls.json", "r", encoding="utf-8") as f:
        urls = json.load(f)

    item = urls["gZ-37"]
    url = item["url"]
    expected_size = item["expected_size"]
    archive_path = Path("datasets/external_dmd/original/dmd-dataset-distraction-gZ-37.tar.gz")
    output_dir = Path("datasets/external_dmd/original/extracted")
    manifest_path = Path("datasets/external_dmd/manifests/dmd_sources.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Resuming download with curl (rate limited to 8 MB/s) to {archive_path}...", flush=True)
    curl_cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--limit-rate", "6M",
        "--retry", "5",
        "--retry-delay", "3",
        "-o", str(archive_path),
        url,
    ]

    import time
    max_retries = 50
    retries = 0
    while True:
        actual_size = archive_path.stat().st_size if archive_path.exists() else 0
        if actual_size == expected_size:
            print(f"Download complete: {actual_size} bytes matching expected size.", flush=True)
            break
        if actual_size > expected_size:
            raise ValueError(f"File size {actual_size} exceeded expected size {expected_size}")

        print(f"Running curl resume at {actual_size}/{expected_size} ({actual_size/expected_size*100:.1f}%)...", flush=True)
        res = subprocess.run(curl_cmd)
        new_size = archive_path.stat().st_size if archive_path.exists() else 0
        if new_size == expected_size:
            print(f"Download complete! {new_size} bytes.", flush=True)
            break
        
        retries += 1
        if retries > max_retries:
            raise RuntimeError(f"Exceeded max retries ({max_retries}) for download.")
        print(f"curl exited with code {res.returncode} (downloaded {new_size} bytes). Resuming in 3 seconds (retry {retries}/{max_retries})...", flush=True)
        time.sleep(3)

    print("Computing SHA256 of downloaded archive...", flush=True)
    archive_sha256 = sha256_file(archive_path)
    print(f"Archive SHA256: {archive_sha256}", flush=True)

    print("Extracting target members (s2 + RGB + BODY)...", flush=True)
    extracted_video_path = None
    extracted_ann_path = None

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            name_lower = member.name.lower()
            if "s2" in name_lower and "rgb_body.mp4" in name_lower:
                dest = output_dir / Path(member.name).name
                print(f"  Extracting video member: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_video_path = dest
            elif "s2" in name_lower and "rgb_ann_distraction.json" in name_lower:
                dest = output_dir / Path(member.name).name
                print(f"  Extracting annotation member: {member.name} -> {dest.name}", flush=True)
                with tar.extractfile(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_ann_path = dest

    if not extracted_video_path or not extracted_video_path.exists():
        raise RuntimeError("Target s2 rgb_body.mp4 member not found in archive")
    if not extracted_ann_path or not extracted_ann_path.exists():
        raise RuntimeError("Target s2 rgb_ann_distraction.json member not found in archive")

    video_size = extracted_video_path.stat().st_size
    video_sha256 = sha256_file(extracted_video_path)
    ann_size = extracted_ann_path.stat().st_size
    ann_sha256 = sha256_file(extracted_ann_path)

    print(f"Extracted video: {extracted_video_path.name} ({video_size} bytes, SHA256: {video_sha256})", flush=True)
    print(f"Extracted annotation: {extracted_ann_path.name} ({ann_size} bytes, SHA256: {ann_sha256})", flush=True)

    manifest_record = {
        "source_id": SOURCE_ID,
        "source_group": "gZ",
        "source_participant_id": "37",
        "source_session_id": "s2",
        "source_channel": "RGB",
        "source_stream": "BODY",
        "archive_filename": archive_path.name,
        "archive_size_bytes": actual_size,
        "archive_sha256": archive_sha256,
        "archive_retained": False,
        "member_path": str(extracted_video_path).replace("\\", "/"),
        "member_size_bytes": video_size,
        "member_sha256": video_sha256,
        "annotation_member_path": str(extracted_ann_path).replace("\\", "/"),
        "annotation_size_bytes": ann_size,
        "annotation_sha256": ann_sha256,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
        "license_reference": LICENSE_REF,
        "dataset_role": DATASET_ROLE,
    }

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")
    print(f"Appended source provenance record to {manifest_path}", flush=True)

    # Clean up archive
    archive_path.unlink(missing_ok=True)
    progress_file = archive_path.with_name(archive_path.name + ".progress.json")
    progress_file.unlink(missing_ok=True)
    print("Archive cleaned up to preserve disk space.", flush=True)


if __name__ == "__main__":
    main()
