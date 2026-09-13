"""Extract s2 video and annotation members for Subject 37 as soon as they are complete."""
from __future__ import annotations

import json
import shutil
import tarfile
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.common import sha256_file
from training.dmd.adapter import parse_dmd_openlabel

archive_path = Path("datasets/external_dmd/original/dmd-dataset-distraction-gZ-37.tar.gz")
output_dir = Path("datasets/external_dmd/original/extracted")
manifest_path = Path("datasets/external_dmd/manifests/dmd_sources.jsonl")
output_dir.mkdir(parents=True, exist_ok=True)
manifest_path.parent.mkdir(parents=True, exist_ok=True)

print("Monitoring archive for s2 members completeness...", flush=True)

while True:
    if not archive_path.exists():
        print("Archive not found yet. Waiting...", flush=True)
        time.sleep(5)
        continue

    try:
        video_member = None
        ann_member = None
        with tarfile.open(archive_path, "r:*") as tar:
            while True:
                try:
                    m = tar.next()
                    if m is None:
                        break
                except Exception:
                    break
                name_lower = m.name.lower()
                if "s2" in name_lower and "rgb_body.mp4" in name_lower:
                    video_member = m
                elif "s2" in name_lower and "rgb_ann_distraction.json" in name_lower:
                    ann_member = m

                if video_member is not None and ann_member is not None:
                    break

            if video_member is not None and ann_member is not None:
                print(f"Found target members: {video_member.name} ({video_member.size} bytes), {ann_member.name} ({ann_member.size} bytes).", flush=True)
                dest_video = output_dir / Path(video_member.name).name
                dest_ann = output_dir / Path(ann_member.name).name
                temp_video = dest_video.with_name(dest_video.name + ".tmp")

                print(f"Extracting video to {temp_video.name}...", flush=True)
                with tar.extractfile(video_member) as src, temp_video.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                if temp_video.stat().st_size == video_member.size:
                    temp_video.replace(dest_video)
                    print(f"Extracted video successfully: {dest_video.name} ({dest_video.stat().st_size} bytes)", flush=True)

                    with tar.extractfile(ann_member) as src, dest_ann.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"Extracted annotation successfully: {dest_ann.name} ({dest_ann.stat().st_size} bytes)", flush=True)

                    # Compute SHA256
                    print("Computing SHA256 hashes...", flush=True)
                    v_sha = sha256_file(dest_video)
                    a_sha = sha256_file(dest_ann)
                    print(f"Video SHA256: {v_sha}", flush=True)
                    print(f"Annotation SHA256: {a_sha}", flush=True)

                    # Validate video
                    cap = cv2.VideoCapture(str(dest_video))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    print(f"Verified video: {total_f} frames, {fps:.2f} fps ({total_f/fps:.1f}s)", flush=True)

                    # Validate annotation
                    ann = parse_dmd_openlabel(dest_ann, fps=fps, total_frames=total_f)
                    print(f"Verified annotation: {len(ann.phone_intervals)} phone intervals found.", flush=True)
                    for pi in ann.phone_intervals:
                        print(f"  - {pi.action}: {pi.start_seconds:.2f}s -> {pi.end_seconds:.2f}s ({pi.duration_seconds:.2f}s)", flush=True)

                    # Check if already in manifest
                    existing_entries = []
                    if manifest_path.exists():
                        for line in manifest_path.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                existing_entries.append(json.loads(line.strip()))

                    has_sub37 = any(str(e.get("source_participant_id")) == "37" for e in existing_entries)
                    if not has_sub37:
                        manifest_record = {
                            "source_id": "EXTERNAL_DMD_VICOMTECH",
                            "source_group": "gZ",
                            "source_participant_id": "37",
                            "source_session_id": "s2",
                            "source_channel": "RGB",
                            "source_stream": "BODY",
                            "archive_filename": archive_path.name,
                            "archive_size_bytes": 4882642095,
                            "archive_sha256": "43e1cad07ce839bad6fd79b8399629d1e86e0bb8951571c3a9fb1e57d0087929",
                            "archive_retained": False,
                            "member_path": str(dest_video).replace("\\", "/"),
                            "member_size_bytes": dest_video.stat().st_size,
                            "member_sha256": v_sha,
                            "annotation_member_path": str(dest_ann).replace("\\", "/"),
                            "annotation_size_bytes": dest_ann.stat().st_size,
                            "annotation_sha256": a_sha,
                            "downloaded_at": datetime.now(timezone.utc).isoformat(),
                            "source_reference": "https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/",
                            "license_reference": "Vicomtech DMD Research License (Non-commercial academic research)",
                            "dataset_role": "FINAL_UNTOUCHED_DEVELOPMENT_HOLDOUT",
                        }
                        with open(manifest_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(manifest_record) + "\n")
                        print(f"Appended Subject 37 record to {manifest_path}", flush=True)

                    print("EXTRACTION_AND_VALIDATION_COMPLETE", flush=True)
                    break
                else:
                    print(f"Video extraction incomplete: {temp_video.stat().st_size}/{video_member.size}. Retrying...", flush=True)
                    temp_video.unlink(missing_ok=True)
    except Exception as e:
        print(f"Extraction attempt error ({archive_path.stat().st_size / 1e9:.2f} GB): {e}. Checking again in 10s...", flush=True)

    time.sleep(10)
