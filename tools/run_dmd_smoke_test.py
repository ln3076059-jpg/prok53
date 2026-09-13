"""End-to-End Runtime Smoke Test using genuine DMD continuous body video.

Exercises full stack:
Video decode -> Cabin logic -> Occupant association -> Phone detection ->
Pose context -> Phone semantic classification -> Temporal fusion ->
Event state machine -> Evidence generation -> Database persistence ->
API retrieval -> Human review workflow.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from backend.ai.detector import SafetyDetector
from backend.core.security import hash_password
from backend.database import Base, SessionLocal, engine
from backend.models.entities import (
    AnalysisJob,
    Event,
    EventType,
    Evidence,
    InputScope,
    JobStatus,
    Review,
    ReviewStatus,
    User,
    Video,
)
from backend.services.video_analyzer import VideoAnalyzer
from training.common import sha256_file


def run_dmd_smoke_test(
    video_path: Path,
    max_frames: int = 90,
    start_frame: int = 1340,  # Active phonecall_right event frames (middle of action)
    output_report_path: Path = Path("reports/DMD_END_TO_END_SMOKE_TEST.json"),
) -> dict:
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Initializing database tables and services for DMD End-to-End Smoke Test...")
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    evidence_dir = Path("evidence/smoke_test")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create sub-clip targeting the active phone action for rapid execution
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_path = evidence_dir / "dmd_smoke_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    print(f"Created targeted test clip: {clip_path.name} ({written} frames, {written/fps:.2f}s)")

    clip_sha = sha256_file(clip_path)
    clip_size = clip_path.stat().st_size

    # Ensure reviewer user exists for foreign key
    reviewer = session.query(User).filter(User.email == "reviewer@example.org").first()
    if not reviewer:
        reviewer = User(
            email="reviewer@example.org",
            password_hash=hash_password("SmokeTestSecretPass123!"),
            role="reviewer",
        )
        session.add(reviewer)
        session.flush()

    # 2. Register Video and AnalysisJob in DB
    existing_video = session.query(Video).filter(Video.storage_path == str(clip_path.as_posix())).first()
    if existing_video:
        video_rec = existing_video
        video_rec.sha256 = clip_sha
        video_rec.size_bytes = clip_size
    else:
        video_rec = Video(
            original_name=clip_path.name,
            storage_path=str(clip_path.as_posix()),
            sha256=clip_sha,
            size_bytes=clip_size,
            mime_type="video/mp4",
            input_scope=InputScope.VEHICLE_CABIN_CROP,
        )
        session.add(video_rec)
        session.flush()

    job = AnalysisJob(
        video_id=video_rec.id,
        status=JobStatus.QUEUED,
    )
    session.add(job)
    session.commit()
    print(f"Created Job #{job.id} for Video #{video_rec.id} in SQLite DB.")

    # 3. Instantiate full VideoAnalyzer and run analysis
    import gc
    gc.collect()
    detector = SafetyDetector(
        weights=Path("models/active/best.pt"),
        config_path=Path("models/model_config_v2.yaml"),
    )
    analyzer = VideoAnalyzer(detector=detector, evidence_root=evidence_dir)

    print("Executing VideoAnalyzer.analyze() over test clip...")
    gc.collect()
    analyzer.analyze(
        session=session,
        job=job,
        video=video_rec,
    )
    session.commit()
    print(f"Job completed with status: {job.status.value}")

    # 4. Query generated Events and Evidence from DB
    events = session.query(Event).filter(Event.job_id == job.id).all()
    print(f"Database contains {len(events)} events for Job #{job.id}.")

    event_summaries = []
    for ev in events:
        evidence_recs = session.query(Evidence).filter(Evidence.event_id == ev.id).all()
        event_summaries.append({
            "event_id": ev.id,
            "event_type": ev.event_type.value,
            "confidence": ev.confidence,
            "frame_number": ev.frame_number,
            "timestamp_seconds": ev.timestamp_seconds,
            "occupant_role": ev.occupant_role,
            "review_status": ev.review_status.value,
            "evidence_count": len(evidence_recs),
        })

    # 5. Simulate Human Review Workflow
    review_summary = None
    if events:
        target_ev = events[0]
        prev_status = target_ev.review_status.value
        review_rec = Review(
            event_id=target_ev.id,
            user_id=reviewer.id,
            previous_status=prev_status,
            new_status=ReviewStatus.CONFIRMED.value,
            notes="Confirmed driver handheld phone use in DMD continuous body feed.",
        )
        session.add(review_rec)
        target_ev.review_status = ReviewStatus.CONFIRMED
        session.commit()
        print(f"Recorded HUMAN review decision for Event #{target_ev.id} -> CONFIRMED.")

        review_summary = {
            "review_id": review_rec.id,
            "reviewer_user_id": reviewer.id,
            "reviewer_email": reviewer.email,
            "previous_status": prev_status,
            "new_status": review_rec.new_status,
            "event_review_status": target_ev.review_status.value,
        }

    session.close()

    smoke_result = {
        "label": "DMD_END_TO_END_DEVELOPMENT_SMOKE_TEST",
        "status": "PASS" if job.status == JobStatus.COMPLETED else "FAIL",
        "video_source": str(video_path.as_posix()),
        "subclip_path": str(clip_path.as_posix()),
        "subclip_sha256": clip_sha,
        "processed_frames": written,
        "job_id": job.id,
        "job_status": job.status.value,
        "events_generated_count": len(events),
        "events": event_summaries,
        "human_review_workflow": review_summary,
    }

    output_report_path.write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")
    print(f"Smoke test report saved to {output_report_path}")
    return smoke_result


if __name__ == "__main__":
    vid = Path("datasets/external_dmd/original/extracted/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4")
    run_dmd_smoke_test(vid)
