from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date, datetime

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.ai.detector import SafetyDetector
from backend.api.dependencies import current_user
from backend.core.config import get_settings
from backend.core.security import create_access_token, verify_password
from backend.database import SessionLocal, get_db
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
from backend.schemas import (
    EventDetailResponse,
    EventResponse,
    JobResponse,
    LoginRequest,
    ReviewRequest,
    TokenResponse,
    VideoResponse,
)
from backend.services.analysis_queue import BackgroundTaskAnalysisQueue
from backend.services.analysis_worker import InferenceWorker
from backend.services.uploads import safe_upload_name, save_upload
from backend.services.video_analyzer import VideoAnalyzer

router = APIRouter()
settings = get_settings()
detector = SafetyDetector(settings.model_path, settings.model_config_path)
worker = InferenceWorker(SessionLocal, detector, settings.evidence_dir)


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_access_token(user.email))


@router.post("/videos", response_model=VideoResponse, status_code=201, tags=["videos"])
async def upload_video(
    upload: UploadFile = File(...),
    input_scope: InputScope = Form(...),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    safe_name = safe_upload_name(upload.filename or "video")
    video_id = uuid.uuid4().hex
    path = (settings.upload_dir / video_id / safe_name).resolve()
    if settings.upload_dir.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Unsafe upload path")
    size, sha = await save_upload(upload, path, settings.max_upload_mb * 1024 * 1024)
    record = Video(
        id=video_id,
        original_name=safe_name,
        storage_path=str(path),
        sha256=sha,
        size_bytes=size,
        mime_type=upload.content_type or "application/octet-stream",
        input_scope=input_scope,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/videos/{video_id}", response_model=VideoResponse, tags=["videos"])
def video_detail(video_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if not (record := db.get(Video, video_id)):
        raise HTTPException(status_code=404, detail="Video not found")
    return record


@router.post(
    "/videos/{video_id}/analyze", response_model=JobResponse, status_code=202, tags=["analysis"]
)
def analyze_video(
    video_id: str,
    tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    if not detector.available:
        raise HTTPException(status_code=503, detail="Locked model weights are not installed")
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    analyzer = VideoAnalyzer(detector, settings.evidence_dir)
    if not analyzer.supports_scope(video.input_scope):
        raise HTTPException(
            status_code=422,
            detail="Raw traffic scenes require configured local vehicle detector weights.",
        )
    job = AnalysisJob(video_id=video_id)
    db.add(job)
    db.commit()
    db.refresh(job)
    BackgroundTaskAnalysisQueue(tasks, worker.run).enqueue(job.id)
    return job


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["analysis"])
def job_detail(job_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if not (job := db.get(AnalysisJob, job_id)):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/events", response_model=list[EventResponse], tags=["events"])
def events(
    event_type: EventType | None = None,
    review_status: ReviewStatus | None = None,
    video_id: str | None = None,
    min_confidence: float = Query(0, ge=0, le=1),
    max_confidence: float = Query(1, ge=0, le=1),
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    query = select(Event).where(Event.confidence.between(min_confidence, max_confidence))
    if event_type:
        query = query.where(Event.event_type == event_type)
    if review_status:
        query = query.where(Event.review_status == review_status)
    if video_id:
        query = query.where(Event.video_id == video_id)
    if date_from:
        query = query.where(Event.created_at >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        query = query.where(Event.created_at < datetime.combine(date_to, datetime.max.time()))
    return list(db.scalars(query.order_by(Event.created_at.desc()).offset(offset).limit(limit)))


@router.get("/events/{event_id}", response_model=EventDetailResponse, tags=["events"])
def event_detail(event_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if not (event := db.get(Event, event_id)):
        raise HTTPException(status_code=404, detail="Event not found")
    evidence = db.scalar(select(Evidence).where(Evidence.event_id == event_id))
    trace_path = settings.evidence_dir / event_id / "trace.json"
    trace = {}
    if trace_path.is_file():
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            trace = {"evidence_status": "TRACE_UNREADABLE"}
    root_url = f"/events/{event_id}/evidence"
    evidence_payload = {
        "available": evidence is not None,
        "original_url": f"{root_url}/original" if evidence else None,
        "annotated_url": f"{root_url}/annotated" if evidence else None,
        "clip_url": f"{root_url}/clip"
        if (settings.evidence_dir / event_id / "evidence.mp4").is_file()
        else None,
        "trace_url": f"{root_url}/trace" if trace_path.is_file() else None,
        "hashes": trace.get("files", {}),
    }
    history = [
        {
            "previous_status": item.previous_status,
            "new_status": item.new_status,
            "notes": item.notes,
            "reviewer_id": item.user_id,
            "created_at": item.created_at,
        }
        for item in db.scalars(
            select(Review).where(Review.event_id == event_id).order_by(Review.created_at)
        )
    ]
    payload = EventResponse.model_validate(event).model_dump()
    payload.update(
        {
            "vehicle_type": trace.get("vehicle_type", "unknown"),
            "fusion_score": trace.get("fusion_score"),
            "temporal_score": trace.get("temporal_score"),
            "evidence": evidence_payload,
            "evidence_trace": trace,
            "review_history": history,
        }
    )
    return payload


@router.get("/events/{event_id}/evidence/{kind}", tags=["events"])
def event_evidence(
    event_id: str,
    kind: str,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    if not db.get(Event, event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    names = {
        "original": ("original.jpg", "image/jpeg"),
        "annotated": ("annotated.jpg", "image/jpeg"),
        "clip": ("evidence.mp4", "video/mp4"),
        "trace": ("trace.json", "application/json"),
    }
    if kind not in names:
        raise HTTPException(status_code=404, detail="Evidence kind not found")
    filename, media_type = names[kind]
    event_root = (settings.evidence_dir / event_id).resolve()
    path = (event_root / filename).resolve()
    if event_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.patch("/events/{event_id}/review", response_model=EventResponse, tags=["events"])
def review_event(
    event_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if payload.status == ReviewStatus.PENDING:
        raise HTTPException(status_code=422, detail="A review cannot return an event to PENDING")
    if not (event := db.get(Event, event_id)):
        raise HTTPException(status_code=404, detail="Event not found")
    if payload.status == ReviewStatus.CONFIRMED and not db.scalar(
        select(Evidence).where(Evidence.event_id == event.id)
    ):
        raise HTTPException(
            status_code=422,
            detail="An event cannot be confirmed without persisted evidence",
        )
    db.add(
        Review(
            event_id=event.id,
            user_id=user.id,
            previous_status=event.review_status.value,
            new_status=payload.status.value,
            notes=payload.notes,
        )
    )
    event.review_status = payload.status
    db.commit()
    db.refresh(event)
    return event


@router.get("/statistics", tags=["statistics"])
def statistics(db: Session = Depends(get_db), _: User = Depends(current_user)):
    runtime = VideoAnalyzer(detector, settings.evidence_dir).runtime_status()
    analyzed = (
        db.scalar(
            select(func.count())
            .select_from(AnalysisJob)
            .where(AnalysisJob.status == JobStatus.COMPLETED)
        )
        or 0
    )
    rows = db.execute(
        select(Event.event_type, Event.review_status, func.count(Event.id)).group_by(
            Event.event_type, Event.review_status
        )
    ).all()
    by_type = {kind.value: 0 for kind in EventType}
    by_status = {status.value: 0 for status in ReviewStatus}
    for kind, status, count in rows:
        by_type[kind.value] += count
        by_status[status.value] += count
    return {
        "analyzed_videos": analyzed,
        "events_by_type": by_type,
        "events_by_review_status": by_status,
        "model": detector.status(),
        "runtime": runtime,
        "event_metrics_status": "NOT_RUN",
    }


@router.get("/exports/events.csv", tags=["exports"])
def export_events(db: Session = Depends(get_db), _: User = Depends(current_user)):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "event_id",
            "type",
            "timestamp",
            "camera",
            "video",
            "vehicle_context_id",
            "occupant_role",
            "confidence",
            "review_status",
            "model_version",
        ]
    )
    for event in db.scalars(select(Event).order_by(Event.created_at.desc())):
        writer.writerow(
            [
                event.id,
                event.event_type.value,
                event.timestamp_seconds,
                "",
                event.video_id,
                event.vehicle_context_id,
                event.occupant_role,
                f"{event.confidence:.6f}",
                event.review_status.value,
                event.model_version,
            ]
        )
    return Response(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )


@router.get("/health", tags=["system"])
def health():
    model = detector.status()
    runtime = VideoAnalyzer(detector, settings.evidence_dir).runtime_status()
    degraded = not model["available"] or runtime["fusion"]["fusion_fail_closed"]
    return {"status": "degraded" if degraded else "ok", "model": model, "runtime": runtime}
