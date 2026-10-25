"""
Create a longer multi-action test video (dmd_long_multi_action_18s.mp4) from DMD gC-14 / gZ-36
containing mixed driver actions separated by normal driving & posture transitions:
  1) 0.0s - 4.5s: Texting right on lap (LAP_TEXTING / texting_right)
  2) 4.5s - 6.5s: Normal driving (hands on steering wheel, no phone)
  3) 6.5s - 11.0s: Phonecall right at ear (EAR_CALL / phonecall_right)
  4) 11.0s - 13.0s: Normal driving (hands on steering wheel, no phone)
  5) 13.0s - 17.5s: Phonecall left at ear (EAR_CALL / phonecall_left)
And run the upgraded V32 VideoAnalyzer on both dmd_smoke_clip.mp4 and dmd_long_multi_action_18s.mp4
so all detections & events reflect V32 Multi-Modal Calibrated Confidence (> 85%).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from sqlalchemy import delete, select

from backend.ai.detector import SafetyDetector
from backend.core.config import get_settings
from backend.database import SessionLocal
from backend.models.entities import AnalysisJob, Detection, Event, Evidence, InputScope, Video
from backend.services.video_analyzer import VideoAnalyzer
from training.common import sha256_file


def extract_multi_action_clip(gc14_video: Path, gz36_video: Path, dst_video: Path) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    out_fps = 10.0
    out_w, out_h = 960, 540

    # Segments: (video_path, start_s, end_s)
    # 1. gC-14 43.0s-47.0s: phonecall_right (EAR_CALL)
    # 2. gC-14 25.0s-27.0s: normal driving (no phone)
    # 3. gZ-36 33.0s-37.0s: texting_right (LAP_TEXTING)
    # 4. gC-14 25.0s-27.0s: normal driving (no phone)
    # 5. gC-14 140.0s-144.0s: phonecall_left (EAR_CALL)
    segments = [
        (gc14_video, 43.0, 47.0),
        (gc14_video, 25.0, 27.0),
        (gz36_video, 33.0, 37.0),
        (gc14_video, 25.0, 27.0),
        (gc14_video, 140.0, 144.0),
    ]

    writer = cv2.VideoWriter(
        str(dst_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h),
    )
    written = 0
    for src_path, start_s, end_s in segments:
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
        step = max(1, int(round(src_fps / out_fps)))
        start_f = int(start_s * src_fps)
        end_f = int(end_s * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for f_idx in range(start_f, end_f):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (f_idx - start_f) % step == 0:
                resized = cv2.resize(frame, (out_w, out_h))
                writer.write(resized)
                written += 1
        cap.release()
    writer.release()
    print(f"Created {dst_video} with {written} frames ({written / out_fps:.1f}s at {out_fps} fps)")


def reanalyze_videos() -> None:
    settings = get_settings()
    detector = SafetyDetector(settings.model_path, settings.model_config_path)

    gc14_path = Path(
        r"D:\.idea\giangdoantotnghiep\dataset\external_dmd\subjects\gC-14\gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4"
    )
    gz36_path = Path(
        r"D:\.idea\giangdoantotnghiep\dataset\external_dmd\subjects\gZ-36\gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4"
    )
    multi_clip_path = Path(r"D:\.idea\giangdoantotnghiep\data\smoke\dmd_long_multi_action_16s.mp4")
    if gc14_path.is_file() and gz36_path.is_file():
        extract_multi_action_clip(gc14_path, gz36_path, multi_clip_path)

    with SessionLocal() as db:
        # 1. Re-analyze existing dmd_smoke_clip.mp4 with V32 confidence calibration (>85%)
        smoke_vid = db.get(Video, "5b88e5f71ec046189721b6dc902a0ae1")
        if smoke_vid and Path(smoke_vid.storage_path).is_file():
            for old_webm in Path(smoke_vid.storage_path).parent.glob("processed_*.webm"):
                old_webm.unlink(missing_ok=True)
            old_events = list(db.scalars(select(Event).where(Event.video_id == smoke_vid.id)))
            for ev in old_events:
                db.execute(delete(Evidence).where(Evidence.event_id == ev.id))
            db.execute(delete(Event).where(Event.video_id == smoke_vid.id))
            old_jobs = list(db.scalars(select(AnalysisJob).where(AnalysisJob.video_id == smoke_vid.id)))
            for jb in old_jobs:
                db.execute(delete(Detection).where(Detection.job_id == jb.id))
            db.execute(delete(AnalysisJob).where(AnalysisJob.video_id == smoke_vid.id))
            db.commit()

            new_job = AnalysisJob(video_id=smoke_vid.id)
            db.add(new_job)
            db.commit()
            db.refresh(new_job)
            analyzer = VideoAnalyzer(detector, settings.evidence_dir)
            analyzer.analyze(db, new_job, smoke_vid)
            dets = list(
                db.scalars(
                    select(Detection).where(
                        Detection.job_id == new_job.id, Detection.class_name == "phone"
                    )
                )
            )
            evs = list(db.scalars(select(Event).where(Event.video_id == smoke_vid.id)))
            print(
                f"Re-analyzed dmd_smoke_clip.mp4: {len(dets)} phone detections "
                f"(min conf={min((d.confidence for d in dets), default=0):.3f}, "
                f"max conf={max((d.confidence for d in dets), default=0):.3f}), "
                f"{len(evs)} events (confs={[round(e.confidence, 3) for e in evs]})"
            )

        # 2. Register and analyze dmd_long_multi_action_16s.mp4
        if multi_clip_path.is_file():
            vid_id = "dmdlongmultiaction32000000000001"
            existing = db.get(Video, vid_id)
            if existing:
                old_events = list(db.scalars(select(Event).where(Event.video_id == vid_id)))
                for ev in old_events:
                    db.execute(delete(Evidence).where(Evidence.event_id == ev.id))
                db.execute(delete(Event).where(Event.video_id == vid_id))
                old_jobs = list(db.scalars(select(AnalysisJob).where(AnalysisJob.video_id == vid_id)))
                for jb in old_jobs:
                    db.execute(delete(Detection).where(Detection.job_id == jb.id))
                db.execute(delete(AnalysisJob).where(AnalysisJob.video_id == vid_id))
                db.delete(existing)
                db.commit()

            upload_dest = (settings.upload_dir / vid_id / multi_clip_path.name).resolve()
            upload_dest.parent.mkdir(parents=True, exist_ok=True)
            for old_webm in upload_dest.parent.glob("processed_*.webm"):
                old_webm.unlink(missing_ok=True)
            shutil.copy2(multi_clip_path, upload_dest)

            long_vid = Video(
                id=vid_id,
                original_name=multi_clip_path.name,
                storage_path=str(upload_dest),
                sha256=sha256_file(upload_dest),
                size_bytes=upload_dest.stat().st_size,
                mime_type="video/mp4",
                input_scope=InputScope.VEHICLE_CABIN_CROP,
            )
            db.add(long_vid)
            db.commit()

            long_job = AnalysisJob(video_id=vid_id)
            db.add(long_job)
            db.commit()
            db.refresh(long_job)

            analyzer2 = VideoAnalyzer(detector, settings.evidence_dir)
            analyzer2.analyze(db, long_job, long_vid)
            dets2 = list(
                db.scalars(
                    select(Detection).where(
                        Detection.job_id == long_job.id, Detection.class_name == "phone"
                    )
                )
            )
            evs2 = list(db.scalars(select(Event).where(Event.video_id == vid_id)))
            print(
                f"Analyzed {multi_clip_path.name}: {len(dets2)} phone detections "
                f"(min conf={min((d.confidence for d in dets2), default=0):.3f}, "
                f"max conf={max((d.confidence for d in dets2), default=0):.3f}), "
                f"{len(evs2)} events (confs={[round(e.confidence, 3) for e in evs2]}, "
                f"times={[round(e.timestamp_seconds, 2) for e in evs2]})"
            )


if __name__ == "__main__":
    reanalyze_videos()
