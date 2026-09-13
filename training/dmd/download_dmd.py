"""High-performance parallel DMD archive downloader and member extractor.

Uses multi-worker HTTP Range requests with thread-safe pre-allocated file writes,
computes verified archive and member SHA256, extracts ONLY s2 RGB BODY mp4
and matching annotation json, and enforces disk retention policy.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from training.common import sha256_file


SOURCE_ID = "EXTERNAL_DMD_VICOMTECH"
DATASET_ROLE = "TEMPORAL_DEVELOPMENT"
LICENSE_REF = "Vicomtech DMD Research License (Non-commercial academic research)"


def download_chunk(
    url: str,
    shared_f: BinaryIO,
    file_lock: threading.Lock,
    start: int,
    end: int,
    chunk_id: int,
    progress_cb: Any,
    on_chunk_done: Any,
    max_retries: int = 10,
) -> int:
    """Download a specific byte range with strict byte count and retries."""
    expected = end - start + 1
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Roadwatch-DMD-Downloader/2.0",
        },
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status not in (200, 206):
                    raise IOError(f"HTTP {resp.status} for range {start}-{end}")

                pos = start
                remaining = expected
                buf_size = 128 * 1024

                while remaining > 0:
                    to_read = min(remaining, buf_size)
                    data = resp.read(to_read)
                    if not data:
                        break
                    with file_lock:
                        shared_f.seek(pos)
                        shared_f.write(data)
                    pos += len(data)
                    remaining -= len(data)
                    progress_cb(len(data))

                if remaining == 0:
                    on_chunk_done(chunk_id)
                    return expected

                print(f"Warning: chunk {chunk_id} incomplete ({expected - remaining}/{expected}), retrying attempt {attempt+1}...", flush=True)
        except Exception as e:
            if attempt == max_retries - 1:
                raise IOError(f"Failed chunk {chunk_id} ({start}-{end}) after {max_retries} attempts: {e}") from e
            time.sleep(min(2.0 ** attempt, 15.0))

    raise IOError(f"Failed chunk {chunk_id} ({start}-{end})")


def parallel_download_file(
    url: str,
    output_path: Path,
    total_size: int,
    num_workers: int = 32,
    chunk_size_mb: int = 2,
) -> str:
    """Download file using parallel range requests with resume support and return its SHA256."""
    chunk_bytes = chunk_size_mb * 1024 * 1024
    num_chunks = (total_size + chunk_bytes - 1) // chunk_bytes
    progress_file = output_path.with_name(output_path.name + ".progress.json")

    completed_chunks: set[int] = set()
    if output_path.exists() and output_path.stat().st_size == total_size and progress_file.exists():
        try:
            completed_chunks = set(json.loads(progress_file.read_text(encoding="utf-8")))
            print(f"Found existing download with {len(completed_chunks)}/{num_chunks} completed chunks. Resuming...", flush=True)
        except Exception:
            completed_chunks = set()
    else:
        print(f"Creating empty pre-allocated file: {output_path.name} ({total_size/(1024**3):.2f} GB)...", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            f.seek(total_size - 1)
            f.write(b"\0")
        completed_chunks = set()

    all_chunks = []
    for i in range(num_chunks):
        c_start = i * chunk_bytes
        c_end = min(total_size - 1, c_start + chunk_bytes - 1)
        all_chunks.append((c_start, c_end, i))

    downloaded_total = sum((end - start + 1) for start, end, cid in all_chunks if cid in completed_chunks)
    remaining_chunks = [c for c in all_chunks if c[2] not in completed_chunks]

    t0 = time.time()
    last_log = t0
    progress_lock = threading.Lock()
    file_lock = threading.Lock()
    state_lock = threading.Lock()

    def on_bytes(n: int) -> None:
        nonlocal downloaded_total, last_log
        with progress_lock:
            downloaded_total += n
            now = time.time()
            if now - last_log >= 10.0:
                elapsed = max(now - t0, 0.1)
                speed = (downloaded_total / (1024 * 1024)) / elapsed
                pct = (downloaded_total / total_size) * 100
                eta_s = (total_size - downloaded_total) / max(speed * 1024 * 1024, 1)
                print(
                    f"  [{now - t0:.0f}s] {downloaded_total/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB "
                    f"({pct:.1f}%) | Speed: {speed:.2f} MB/s | ETA: {eta_s:.0f}s",
                    flush=True,
                )
                last_log = now

    def on_chunk_done(cid: int) -> None:
        with state_lock:
            completed_chunks.add(cid)
            try:
                progress_file.write_text(json.dumps(sorted(completed_chunks)), encoding="utf-8")
            except Exception:
                pass

    if remaining_chunks:
        print(f"Starting parallel download with {num_workers} workers over {len(remaining_chunks)}/{num_chunks} chunks...", flush=True)
        with output_path.open("r+b") as shared_f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        download_chunk,
                        url,
                        shared_f,
                        file_lock,
                        start,
                        end,
                        cid,
                        on_bytes,
                        on_chunk_done,
                    )
                    for start, end, cid in remaining_chunks
                ]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
            shared_f.flush()
    else:
        print("All chunks already downloaded!", flush=True)

    progress_file.unlink(missing_ok=True)
    total_elapsed = max(time.time() - t0, 0.1)
    avg_speed = (total_size / (1024 * 1024)) / total_elapsed
    print(f"Download complete: {total_size} bytes in {total_elapsed:.1f}s (Avg {avg_speed:.2f} MB/s)", flush=True)

    print("Computing SHA256 of downloaded archive...", flush=True)
    h = hashlib.sha256()
    with output_path.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    sha_val = h.hexdigest()
    print(f"Archive SHA256: {sha_val}", flush=True)
    return sha_val


def download_and_extract_dmd(
    url: str,
    archive_filename: str,
    output_dir: Path,
    manifest_path: Path,
    cleanup_archive: bool = True,
    expected_size: int | None = None,
    num_workers: int = 32,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir.parent / archive_filename

    # Query content-length if not provided
    if not expected_size:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as resp:
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr:
                expected_size = int(cr.split("/")[-1])
            else:
                expected_size = int(resp.headers.get("Content-Length", 0))

    if not expected_size:
        raise ValueError("Could not determine archive total size")

    # Check free disk space before download
    disk_usage = shutil.disk_usage(archive_path.parent)
    free_gb = disk_usage.free / (1024**3)
    needed_gb = (expected_size / (1024**3)) + 1.5
    print(f"Target disk: {free_gb:.2f} GB free (requires ~{needed_gb:.2f} GB)", flush=True)
    if free_gb < needed_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB < {needed_gb:.2f} GB")

    archive_sha256 = parallel_download_file(
        url=url,
        output_path=archive_path,
        total_size=expected_size,
        num_workers=num_workers,
    )

    # Inspect tar archive and extract only target members
    import tarfile

    extracted_video_path: Path | None = None
    extracted_ann_path: Path | None = None
    member_paths: list[str] = []

    print(f"Extracting target members (s2 + RGB + BODY) from {archive_filename}...", flush=True)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            member_paths.append(member.name)
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
        raise RuntimeError(f"Target s2 rgb_body.mp4 member not found in archive: {archive_filename}")
    if not extracted_ann_path or not extracted_ann_path.exists():
        raise RuntimeError(f"Target s2 rgb_ann_distraction.json member not found in archive: {archive_filename}")

    video_size = extracted_video_path.stat().st_size
    video_sha256 = sha256_file(extracted_video_path)
    ann_size = extracted_ann_path.stat().st_size
    ann_sha256 = sha256_file(extracted_ann_path)

    print(f"Extracted video: {extracted_video_path.name} ({video_size} bytes, SHA256: {video_sha256})", flush=True)
    print(f"Extracted annotation: {extracted_ann_path.name} ({ann_size} bytes, SHA256: {ann_sha256})", flush=True)

    parts = extracted_video_path.name.split("_")
    group = parts[0] if len(parts) > 0 else "unknown"
    participant = parts[1] if len(parts) > 1 else "unknown"
    session = parts[2] if len(parts) > 2 else "s2"

    manifest_record = {
        "source_id": SOURCE_ID,
        "source_group": group,
        "source_participant_id": participant,
        "source_session_id": session,
        "source_channel": "RGB",
        "source_stream": "BODY",
        "archive_filename": archive_filename,
        "archive_size_bytes": expected_size,
        "archive_sha256": archive_sha256,
        "archive_retained": not cleanup_archive,
        "member_path": str(extracted_video_path.as_posix()),
        "member_size_bytes": video_size,
        "member_sha256": video_sha256,
        "annotation_member_path": str(extracted_ann_path.as_posix()),
        "annotation_size_bytes": ann_size,
        "annotation_sha256": ann_sha256,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
        "license_reference": LICENSE_REF,
        "dataset_role": DATASET_ROLE,
    }

    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")
    print(f"Recorded manifest entry for subject {participant} in {manifest_path}", flush=True)

    if cleanup_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Retention policy: removed archive {archive_filename} to preserve disk space.", flush=True)

    return manifest_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel download and extract DMD members.")
    parser.add_argument("--url", required=True, help="Presigned download URL")
    parser.add_argument("--archive", required=True, help="Archive filename")
    parser.add_argument("--expected-size", type=int, default=None, help="Expected byte size")
    parser.add_argument("--output-dir", default="datasets/external_dmd/original/extracted", help="Extraction directory")
    parser.add_argument("--manifest", default="datasets/external_dmd/manifests/dmd_sources.jsonl", help="Manifest path")
    parser.add_argument("--keep-archive", action="store_true", help="Retain archive on disk")
    parser.add_argument("--workers", type=int, default=16, help="Number of download workers")

    args = parser.parse_args()
    download_and_extract_dmd(
        url=args.url,
        archive_filename=args.archive,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        cleanup_archive=not args.keep_archive,
        expected_size=args.expected_size,
        num_workers=args.workers,
    )
