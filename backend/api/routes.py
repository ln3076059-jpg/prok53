from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date, datetime
from pathlib import Path

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
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from backend.ai.detector import SafetyDetector
from backend.ai.evidence import evidence_package_sha256, sha256_file
from backend.api.dependencies import current_user, export_user, reviewer_user
from backend.core.config import get_settings
from backend.core.security import create_access_token, verify_password
from backend.database import SessionLocal, get_db
from backend.models.entities import (
    AnalysisJob,
    Detection,
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
    PhoneResearchReviewRequest,
    ReviewRequest,
    TokenResponse,
    VideoResponse,
)
from backend.services.analysis_queue import BackgroundTaskAnalysisQueue, DurableAnalysisQueue
from backend.services.analysis_worker import InferenceWorker
from backend.services.research_review import (
    ResearchReviewConflict,
    ResearchReviewError,
    list_phone_reviews,
    phone_review_contact_sheet,
    phone_review_detail,
    save_phone_review,
)
from backend.services.uploads import safe_upload_name, save_upload
from backend.services.v32_benchmark import (
    get_v32_benchmark_payload,
    get_v32_summary_for_stats,
    seed_v32_demo_events,
)
from backend.services.video_analyzer import VideoAnalyzer

router = APIRouter()
settings = get_settings()
detector = SafetyDetector(settings.model_path, settings.model_config_path)
worker = InferenceWorker(SessionLocal, detector, settings.evidence_dir)


def _csv_safe(value):
    if not isinstance(value, str):
        return value
    if value[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def _research_review_error(exc: ResearchReviewError) -> HTTPException:
    status = 409 if isinstance(exc, ResearchReviewConflict) else 422
    return HTTPException(status_code=status, detail=str(exc))


def _event_evidence_root(event_id: str) -> Path:
    base = settings.evidence_dir.resolve()
    root = (base / event_id).resolve()
    if base not in root.parents:
        raise HTTPException(status_code=400, detail="Unsafe event evidence path")
    return root


def _read_evidence_trace(event_id: str) -> tuple[dict, Path]:
    trace_path = _event_evidence_root(event_id) / "trace.json"
    if not trace_path.is_file():
        return {}, trace_path
    try:
        return json.loads(trace_path.read_text(encoding="utf-8")), trace_path
    except (OSError, json.JSONDecodeError):
        return {"evidence_status": "TRACE_UNREADABLE"}, trace_path


def _evidence_integrity(event_id: str, evidence: Evidence | None, trace: dict) -> dict:
    root = _event_evidence_root(event_id)
    original = (
        root / "original_keyframe.jpg"
        if (root / "original_keyframe.jpg").is_file()
        else root / "original.jpg"
    )
    annotated = (
        root / "annotated_keyframe.jpg"
        if (root / "annotated_keyframe.jpg").is_file()
        else root / "annotated.jpg"
    )
    expected = {
        original.name: original,
        annotated.name: annotated,
        "evidence.mp4": root / "evidence.mp4",
        "trace.json": root / "trace.json",
    }
    errors: list[str] = []
    hashes = {name: sha256_file(path) for name, path in expected.items() if path.is_file()}
    integrity_anchor = "MISSING"
    package_sha256 = evidence_package_sha256(hashes)
    if evidence is None:
        errors.append("evidence database record is missing")
    else:
        if Path(evidence.original_path).resolve() != original:
            errors.append("original evidence path does not match the event directory")
        if Path(evidence.annotated_path).resolve() != annotated:
            errors.append("annotated evidence path does not match the event directory")
        if evidence.sha256 == package_sha256:
            integrity_anchor = "FULL_PACKAGE_SHA256"
        elif hashes.get(original.name) == evidence.sha256:
            integrity_anchor = "LEGACY_ORIGINAL_ONLY"
        else:
            integrity_anchor = "INVALID"
            errors.append("evidence package SHA-256 does not match the database record")
    for name, path in (("original keyframe", original), ("annotated keyframe", annotated)):
        if not path.is_file():
            errors.append(f"required evidence file is missing: {name}")
    if "trace.json" not in hashes:
        errors.append("required evidence file is missing: trace.json")
    for name, declared_hash in trace.get("files", {}).items():
        if name in hashes and hashes[name] != declared_hash:
            errors.append(f"evidence SHA-256 mismatch: {name}")

    available = not errors
    confirmation_blockers = list(errors)
    if integrity_anchor != "FULL_PACKAGE_SHA256":
        confirmation_blockers.append("evidence package is not anchored by a full package SHA-256")
    if trace.get("evidence_status") != "COMPLETE":
        confirmation_blockers.append("temporal evidence clip is not complete")
    if "evidence.mp4" not in hashes:
        confirmation_blockers.append("temporal evidence clip is missing")
    return {
        "available": available,
        "confirmation_ready": available and not confirmation_blockers,
        "integrity_status": (
            "MISSING" if evidence is None else "VALID" if not errors else "INVALID"
        ),
        "integrity_anchor": integrity_anchor,
        "package_sha256": package_sha256,
        "integrity_errors": errors,
        "confirmation_blockers": confirmation_blockers,
        "hashes": hashes,
    }


def _commit_review_transition(
    db: Session, event: Event, user: User, payload: ReviewRequest
) -> Event:
    """Append a review and change event state only if its observed state is still current."""
    previous_status = event.review_status
    transition = db.execute(
        update(Event)
        .where(Event.id == event.id, Event.review_status == previous_status)
        .values(review_status=payload.status)
    )
    if transition.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=409, detail="Review state changed concurrently")
    db.add(
        Review(
            event_id=event.id,
            user_id=user.id,
            previous_status=previous_status.value,
            new_status=payload.status.value,
            notes=payload.notes,
        )
    )
    db.commit()
    db.refresh(event)
    return event


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


@router.get("/videos", tags=["videos"])
def list_videos(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    videos = list(db.scalars(select(Video).order_by(Video.created_at.desc()).limit(limit)))
    results = []
    for v in videos:
        storage = Path(v.storage_path) if not str(v.storage_path).startswith("benchmark://") else None
        streamable = bool(storage and storage.is_file())
        latest_job = db.scalar(
            select(AnalysisJob)
            .where(AnalysisJob.video_id == v.id)
            .order_by(AnalysisJob.created_at.desc())
        )
        event_count = (
            db.scalar(select(func.count()).select_from(Event).where(Event.video_id == v.id)) or 0
        )
        detection_count = 0
        if latest_job:
            detection_count = (
                db.scalar(
                    select(func.count())
                    .select_from(Detection)
                    .where(Detection.job_id == latest_job.id)
                )
                or 0
            )
        results.append(
            {
                "id": v.id,
                "original_name": v.original_name,
                "size_bytes": v.size_bytes,
                "mime_type": v.mime_type,
                "input_scope": v.input_scope.value,
                "created_at": v.created_at,
                "streamable": streamable,
                "event_count": event_count,
                "detection_count": detection_count,
                "latest_job": (
                    {
                        "id": latest_job.id,
                        "status": latest_job.status.value,
                        "progress": latest_job.progress,
                        "error": latest_job.error,
                        "completed_at": latest_job.completed_at,
                    }
                    if latest_job
                    else None
                ),
            }
        )
    return results


@router.get("/videos/{video_id}", response_model=VideoResponse, tags=["videos"])
def video_detail(video_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if not (record := db.get(Video, video_id)):
        raise HTTPException(status_code=404, detail="Video not found")
    return record


import shutil
import threading

_VIDEO_RENDER_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


def _get_video_render_lock(video_id: str) -> threading.Lock:
    with _LOCK_GUARD:
        if video_id not in _VIDEO_RENDER_LOCKS:
            _VIDEO_RENDER_LOCKS[video_id] = threading.Lock()
        return _VIDEO_RENDER_LOCKS[video_id]


def _resolve_and_copy_external_video(video: Video, db: Session | None = None) -> Path | None:
    """If a video resides in an outside directory (thu muc ngoai), copy it into uploads/{video.id}/ (thu muc can)."""
    raw_str = str(video.storage_path or "")
    if not raw_str or raw_str.startswith("benchmark://"):
        return None
    target_dir = Path("uploads") / video.id
    safe_name = Path(video.original_name or Path(raw_str).name).name
    if not safe_name.lower().endswith((".mp4", ".webm", ".avi", ".mov", ".mkv")):
        safe_name = f"{safe_name}.mp4"
    target_path = target_dir / safe_name

    if target_path.is_file() and target_path.stat().st_size > 256:
        if Path(raw_str).resolve() != target_path.resolve() and db is not None:
            video.storage_path = str(target_path)
            db.commit()
        return target_path

    candidates = [
        Path(raw_str),
        Path("data/smoke") / safe_name,
        Path("data/smoke") / Path(raw_str).name,
    ]
    for ext_file in Path("uploads").rglob(safe_name):
        if ext_file.is_file() and ext_file.stat().st_size > 256:
            candidates.append(ext_file)
            break

    source_file = next((c for c in candidates if c.is_file() and c.stat().st_size > 256), None)
    if source_file is None:
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    if source_file.resolve() != target_path.resolve():
        shutil.copy2(source_file, target_path)
    if db is not None and str(video.storage_path) != str(target_path):
        video.storage_path = str(target_path)
        db.commit()
    return target_path


def _ensure_browser_processed_video(
    video_path: Path, video_id: str, db: Session
) -> tuple[Path, str]:
    """Render a browser-playable WebM (VP80) stream with deduplicated, frame-locked AI bounding boxes so Fullscreen and PiP always keep overlays visible without double labels."""
    import cv2

    video_obj = db.get(Video, video_id)
    if video_obj is not None:
        resolved = _resolve_and_copy_external_video(video_obj, db)
        if resolved is not None:
            video_path = resolved

    latest_job = db.scalar(
        select(AnalysisJob)
        .where(AnalysisJob.video_id == video_id)
        .order_by(AnalysisJob.created_at.desc())
    )
    target_dir = Path("uploads") / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_webm = target_dir / f"annotated_{latest_job.id if latest_job else 'raw'}.webm"
    export_dir = Path("outputs")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_webm = export_dir / f"annotated_{video_id}_{video_path.stem}.webm"

    job_completed_ts = (
        latest_job.completed_at.timestamp()
        if latest_job is not None and latest_job.completed_at is not None
        else 0.0
    )
    if (
        out_webm.is_file()
        and out_webm.stat().st_size > 256
        and out_webm.stat().st_mtime >= job_completed_ts
    ):
        if not export_webm.is_file() or export_webm.stat().st_mtime < out_webm.stat().st_mtime:
            shutil.copy2(out_webm, export_webm)
        return out_webm, "video/webm"

    render_lock = _get_video_render_lock(video_id)
    with render_lock:
        if (
            out_webm.is_file()
            and out_webm.stat().st_size > 256
            and out_webm.stat().st_mtime >= job_completed_ts
        ):
            if not export_webm.is_file() or export_webm.stat().st_mtime < out_webm.stat().st_mtime:
                shutil.copy2(out_webm, export_webm)
            return out_webm, "video/webm"

        seatbelt_by_frame: dict[int, list[Detection]] = {}
        phone_by_frame: dict[int, list[Detection]] = {}
        if latest_job is not None:
            dets = list(
                db.scalars(
                    select(Detection)
                    .where(Detection.job_id == latest_job.id)
                    .order_by(Detection.frame_number.asc(), Detection.confidence.desc())
                )
            )
            seen_frame_role_kind: set[tuple[int, str, str, int]] = set()
            for d in dets:
                role_key = (d.occupant_role or "driver").lower()
                kind_key = "phone" if "phone" in d.class_name.lower() else "seatbelt"
                bdict = d.bbox or {}
                xyxy = bdict.get("xyxy")
                x_ref = float(xyxy[0]) if isinstance(xyxy, list) and len(xyxy) >= 4 else float(bdict.get("x1", 0.0) or 0.0)
                slot_key = int(d.track_id) if d.track_id is not None else int(x_ref // 240.0)
                dedup_key = (d.frame_number, role_key, kind_key, slot_key)
                if dedup_key in seen_frame_role_kind:
                    continue
                seen_frame_role_kind.add(dedup_key)
                if kind_key == "phone":
                    phone_by_frame.setdefault(d.frame_number, []).append(d)
                else:
                    seatbelt_by_frame.setdefault(d.frame_number, []).append(d)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_path, "video/mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        if fps <= 0 or fps > 120:
            fps = 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        frame_stride = 5 if fps >= 24.0 else 3
        out_fps = max(5.0, fps / frame_stride)
        max_out_w = 640
        if width > max_out_w:
            scale = max_out_w / float(width)
            out_w = max_out_w
            out_h = max(2, int(round(height * scale)) & ~1)
        else:
            scale = 1.0
            out_w = width
            out_h = height

        tmp_webm = out_webm.with_name(f"tmp_{uuid.uuid4().hex[:8]}_{out_webm.name}")
        writer = cv2.VideoWriter(
            str(tmp_webm),
            cv2.VideoWriter_fourcc(*"VP80"),
            out_fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            cap.release()
            return video_path, "video/mp4"

        sb_frames = sorted(seatbelt_by_frame.keys())
        ph_frames = sorted(phone_by_frame.keys())
        frame_idx = 0
        while True:
            if frame_idx % frame_stride != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            active: list[Detection] = []
            if sb_frames:
                nearest_sb = min(sb_frames, key=lambda fn: abs(fn - frame_idx))
                if abs(nearest_sb - frame_idx) <= 10:
                    active.extend(seatbelt_by_frame[nearest_sb])
            if ph_frames:
                nearest_ph = min(ph_frames, key=lambda fn: abs(fn - frame_idx))
                if abs(nearest_ph - frame_idx) <= 8:
                    active.extend(phone_by_frame[nearest_ph])

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            placed_banners: list[tuple[int, int, int, int]] = []
            phone_reserved: list[tuple[int, int, int, int]] = []
            torso_boxes_scaled: list[tuple[int, int, int, int]] = []
            for det in active:
                bdict = det.bbox or {}
                xy = bdict.get("xyxy")
                if not isinstance(xy, list) or len(xy) < 4:
                    continue
                sx1, sy1, sx2, sy2 = [int(round(float(v) * scale)) for v in xy[:4]]
                if "phone" in (det.class_name or "").lower():
                    phone_reserved.append((max(2, sx1 - 6), max(2, sy1 - 26), min(out_w - 2, sx2 + 120), sy2 + 4))
                else:
                    torso_boxes_scaled.append((sx1, sy1, sx2, sy2))
            torso_boxes_scaled.sort(key=lambda b: b[0])

            # Sort active detections so left-to-right torso boxes place first, then phone boxes
            active_sorted = sorted(
                active,
                key=lambda d: (
                    1 if "phone" in (d.class_name or "").lower() else 0,
                    float(((d.bbox or {}).get("xyxy") or [0])[0]),
                ),
            )

            torso_idx = 0
            for idx_d, det in enumerate(active_sorted):
                bdict = det.bbox or {}
                xyxy = bdict.get("xyxy")
                if (not isinstance(xyxy, list) or len(xyxy) < 4) and all(
                    k in bdict for k in ("x1", "y1", "x2", "y2")
                ):
                    xyxy = [bdict["x1"], bdict["y1"], bdict["x2"], bdict["y2"]]
                if not isinstance(xyxy, list) or len(xyxy) < 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in xyxy[:4]]
                x1 = max(2, min(out_w - 4, x1))
                y1 = max(2, min(out_h - 4, y1))
                x2 = max(x1 + 4, min(out_w - 2, x2))
                y2 = max(y1 + 4, min(out_h - 2, y2))

                cname = det.class_name.lower()
                role = (det.occupant_role or "driver").lower()
                is_driver = role == "driver"
                role_prefix = "DRIVER" if is_driver else "PASSENGER"
                is_phone = "phone" in cname
                is_unfast = "unfastened" in cname or "no_seatbelt" in cname

                if is_phone and is_driver:
                    color = (40, 30, 218)  # Red (VI PHAM)
                    tag = f"{role_prefix} - PHONE [VI PHAM]"
                elif is_phone and not is_driver:
                    color = (195, 114, 0)  # Blue (KHONG VI PHAM)
                    tag = f"{role_prefix} - PHONE [KHONG VI PHAM]"
                elif is_unfast:
                    color = (43, 131, 255)  # Orange (VI PHAM)
                    tag = f"{role_prefix} - NO_SEATBELT [VI PHAM]"
                else:
                    color = (72, 161, 36)  # Green (OK)
                    tag = f"{role_prefix} - SEATBELT [OK]"

                posture = (det.bbox or {}).get("phone_posture")
                posture_str = f" ({posture})" if (is_phone and is_driver and posture) else ""
                label = f"{tag} {int(round(det.confidence * 100))}%{posture_str}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                font_sc = 0.40
                font_th = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_sc, font_th)
                bh = th + 8
                bw = tw + 10
                lx = max(4, min(out_w - bw - 4, x1))
                if not is_phone and len(torso_boxes_scaled) >= 2 and torso_idx == 0:
                    right_x1 = torso_boxes_scaled[1][0]
                    if lx + bw > right_x1 - 4:
                        lx = max(4, min(x1, right_x1 - bw - 6))
                top_y = max(2, y1 - bh)
                bot_y = max(bh, min(out_h - 2 - bh, y2 - bh - 2))
                below_y = min(out_h - 2 - bh, y2 + 2)
                if not is_phone and len(torso_boxes_scaled) >= 2 and torso_idx == 1:
                    # For the second (right) occupant in a 2-person shot:
                    # If top_y would overlap horizontally with the left occupant's top banner or a phone banner,
                    # prefer bottom of the box so it never covers the right occupant's face.
                    candidates = [
                        (lx, top_y, lx + bw, top_y + bh),
                        (lx, bot_y, lx + bw, bot_y + bh),
                        (lx, below_y, lx + bw, below_y + bh),
                    ]
                elif not is_phone:
                    candidates = [
                        (lx, top_y, lx + bw, top_y + bh),
                        (lx, below_y, lx + bw, below_y + bh),
                        (lx, bot_y, lx + bw, bot_y + bh),
                    ]
                else:
                    candidates = [
                        (lx, top_y, lx + bw, top_y + bh),
                        (lx, below_y, lx + bw, below_y + bh),
                        (lx, bot_y, lx + bw, bot_y + bh),
                    ]
                if not is_phone:
                    torso_idx += 1

                obstacles = placed_banners if is_phone else (placed_banners + phone_reserved)
                chosen = candidates[0]
                for cx1, cy1, cx2, cy2 in candidates:
                    overlaps = any(
                        not (cx2 + 4 <= ox1 or cx1 - 4 >= ox2 or cy2 + 2 <= oy1 or cy1 - 2 >= oy2)
                        for ox1, oy1, ox2, oy2 in obstacles
                    )
                    if not overlaps:
                        chosen = (cx1, cy1, cx2, cy2)
                        break
                placed_banners.append(chosen)
                bx1, by1, bx2, by2 = chosen
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (bx1 + 5, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_sc,
                    (255, 255, 255),
                    font_th,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1

        writer.release()
        cap.release()
        if frame_idx > 0 and tmp_webm.is_file() and tmp_webm.stat().st_size > 256:
            tmp_webm.replace(out_webm)
            shutil.copy2(out_webm, export_webm)
            return out_webm, "video/webm"
        if tmp_webm.is_file():
            tmp_webm.unlink(missing_ok=True)
        return video_path, "video/mp4"


def _ensure_browser_evidence_clip(mp4_path: Path) -> tuple[Path, str]:
    """Transcode OpenCV mp4v evidence.mp4 into VP80 WebM so HTML5 <video> can play it."""
    import cv2

    webm_path = mp4_path.parent / "evidence_browser.webm"
    mp4_mtime = mp4_path.stat().st_mtime if mp4_path.is_file() else 0.0
    if (
        webm_path.is_file()
        and webm_path.stat().st_size > 256
        and webm_path.stat().st_mtime >= mp4_mtime
    ):
        return webm_path, "video/webm"

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return mp4_path, "video/mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    if fps <= 0 or fps > 120:
        fps = 8.0
    total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if 0 < total_f <= 35 and fps >= 15.0:
        fps = max(4.5, fps / 5.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return mp4_path, "video/mp4"

    writer = cv2.VideoWriter(
        str(webm_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return mp4_path, "video/mp4"

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written > 0 and webm_path.is_file() and webm_path.stat().st_size > 256:
        return webm_path, "video/webm"
    return mp4_path, "video/mp4"


@router.get("/videos/{video_id}/stream", tags=["videos"])
def stream_video(video_id: str, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(video.storage_path).startswith("benchmark://"):
        raise HTTPException(status_code=404, detail="Synthetic benchmark video has no local stream")
    resolved = _resolve_and_copy_external_video(video, db)
    path = (resolved if resolved is not None else Path(video.storage_path)).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    stream_path, media_type = _ensure_browser_processed_video(path, video_id, db)
    return FileResponse(
        stream_path,
        media_type=media_type,
        filename=stream_path.name,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/videos/{video_id}/analysis-summary", tags=["videos"])
def video_analysis_summary(
    video_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    resolved = _resolve_and_copy_external_video(video, db)
    storage = resolved if resolved is not None else (Path(video.storage_path) if not str(video.storage_path).startswith("benchmark://") else None)
    streamable = bool(storage and storage.is_file())
    latest_job = db.scalar(
        select(AnalysisJob)
        .where(AnalysisJob.video_id == video_id)
        .order_by(AnalysisJob.created_at.desc())
    )
    video_events = list(
        db.scalars(
            select(Event)
            .where(Event.video_id == video_id)
            .order_by(Event.timestamp_seconds.asc(), Event.frame_number.asc())
        )
    )
    detailed_events = []
    for ev in video_events:
        evidence = db.scalar(select(Evidence).where(Evidence.event_id == ev.id))
        trace, trace_path = _read_evidence_trace(ev.id)
        integrity = _evidence_integrity(ev.id, evidence, trace)
        root_url = f"/events/{ev.id}/evidence"
        ev_root = _event_evidence_root(ev.id)
        orig_path = (
            ev_root / "original_keyframe.jpg"
            if (ev_root / "original_keyframe.jpg").is_file()
            else ev_root / "original.jpg"
        )
        annot_path = (
            ev_root / "annotated_keyframe.jpg"
            if (ev_root / "annotated_keyframe.jpg").is_file()
            else ev_root / "annotated.jpg"
        )
        ev_payload = EventResponse.model_validate(ev).model_dump()
        ev_payload.update(
            {
                "vehicle_type": trace.get("vehicle_type", "unknown"),
                "fusion_score": trace.get("fusion_score"),
                "temporal_score": trace.get("temporal_score"),
                "evidence": {
                    **integrity,
                    "original_url": f"{root_url}/original" if orig_path.is_file() else None,
                    "annotated_url": f"{root_url}/annotated" if annot_path.is_file() else None,
                    "clip_url": f"{root_url}/clip" if (ev_root / "evidence.mp4").is_file() else None,
                    "trace_url": f"{root_url}/trace" if trace_path.is_file() else None,
                },
                "evidence_trace": trace,
            }
        )
        detailed_events.append(ev_payload)

    detections_payload = []
    if latest_job:
        det_rows = list(
            db.scalars(
                select(Detection)
                .where(Detection.job_id == latest_job.id)
                .order_by(Detection.timestamp_seconds.asc(), Detection.frame_number.asc())
                .limit(2000)
            )
        )
        for d in det_rows:
            detections_payload.append(
                {
                    "id": d.id,
                    "frame_number": d.frame_number,
                    "timestamp_seconds": d.timestamp_seconds,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox or {},
                    "track_id": d.track_id,
                    "driver_associated": d.driver_associated,
                    "occupant_role": d.occupant_role,
                    "vehicle_context_id": d.vehicle_context_id,
                }
            )

    phone_det_count = sum(1 for d in detections_payload if "phone" in d["class_name"].lower())
    seatbelt_det_count = sum(1 for d in detections_payload if "seatbelt" in d["class_name"].lower())
    occupant_det_count = sum(1 for d in detections_payload if "occupant" in d["class_name"].lower() or "person" in d["class_name"].lower())

    return {
        "video": {
            "id": video.id,
            "original_name": video.original_name,
            "size_bytes": video.size_bytes,
            "mime_type": video.mime_type,
            "input_scope": video.input_scope.value,
            "created_at": video.created_at,
            "streamable": streamable,
        },
        "job": (
            {
                "id": latest_job.id,
                "status": latest_job.status.value,
                "progress": latest_job.progress,
                "error": latest_job.error,
                "created_at": latest_job.created_at,
                "completed_at": latest_job.completed_at,
            }
            if latest_job
            else None
        ),
        "summary": {
            "total_detections": len(detections_payload),
            "phone_detections": phone_det_count,
            "seatbelt_detections": seatbelt_det_count,
            "occupant_detections": occupant_det_count,
            "total_events": len(detailed_events),
            "phone_events": sum(1 for e in detailed_events if e["event_type"] == EventType.PHONE),
            "no_seatbelt_events": sum(
                1 for e in detailed_events if e["event_type"] == EventType.NO_SEATBELT
            ),
        },
        "events": detailed_events,
        "detections": detections_payload,
    }


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
    if settings.analysis_queue_backend == "database":
        DurableAnalysisQueue(SessionLocal).enqueue(job.id)
    else:
        BackgroundTaskAnalysisQueue(tasks, worker.run).enqueue(job.id)
    return job


@router.get("/health/live", tags=["health"])
def liveness():
    return {"status": "LIVE"}


@router.get("/health/ready", tags=["health"])
def readiness():
    model = detector.status()
    ready = model["available"] and not (
        settings.app_env == "production" and model["activation_state"] != "PRODUCTION_APPROVED"
    )
    payload = {
        "status": "READY" if ready else "NOT_READY",
        "queue_backend": settings.analysis_queue_backend,
        "model": model,
    }
    if not ready:
        return Response(content=json.dumps(payload), status_code=503, media_type="application/json")
    return payload


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["analysis"])
def job_detail(job_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if not (job := db.get(AnalysisJob, job_id)):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/research/phone-reviews", tags=["research-review"])
def research_phone_reviews(_: User = Depends(reviewer_user)):
    try:
        return list_phone_reviews(
            settings.research_phone_review_source,
            settings.research_phone_review_dir,
        )
    except ResearchReviewError as exc:
        raise _research_review_error(exc) from exc


@router.get("/research/phone-reviews/{case_id}", tags=["research-review"])
def research_phone_review_detail(case_id: str, _: User = Depends(reviewer_user)):
    try:
        return phone_review_detail(
            settings.research_phone_review_source,
            settings.research_phone_review_dir,
            case_id,
        )
    except ResearchReviewError as exc:
        raise _research_review_error(exc) from exc


@router.get("/research/phone-reviews/{case_id}/contact-sheet", tags=["research-review"])
def research_phone_review_contact_sheet(case_id: str, _: User = Depends(reviewer_user)):
    try:
        path = phone_review_contact_sheet(settings.research_phone_review_source, case_id)
    except ResearchReviewError as exc:
        raise _research_review_error(exc) from exc
    return FileResponse(path, media_type="image/jpeg", filename=f"{case_id}.jpg")


@router.patch("/research/phone-reviews/{case_id}", tags=["research-review"])
def review_research_phone_case(
    case_id: str,
    payload: PhoneResearchReviewRequest,
    user: User = Depends(reviewer_user),
):
    try:
        return save_phone_review(
            settings.research_phone_review_source,
            settings.research_phone_review_dir,
            case_id,
            payload.model_dump(),
            user.id,
        )
    except ResearchReviewError as exc:
        raise _research_review_error(exc) from exc


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
    trace, trace_path = _read_evidence_trace(event_id)
    integrity = _evidence_integrity(event_id, evidence, trace)
    root_url = f"/events/{event_id}/evidence"
    event_root = _event_evidence_root(event_id)
    original_path = (
        event_root / "original_keyframe.jpg"
        if (event_root / "original_keyframe.jpg").is_file()
        else event_root / "original.jpg"
    )
    annotated_path = (
        event_root / "annotated_keyframe.jpg"
        if (event_root / "annotated_keyframe.jpg").is_file()
        else event_root / "annotated.jpg"
    )
    evidence_payload = {
        **integrity,
        "original_url": f"{root_url}/original" if original_path.is_file() else None,
        "annotated_url": f"{root_url}/annotated" if annotated_path.is_file() else None,
        "clip_url": f"{root_url}/clip" if (event_root / "evidence.mp4").is_file() else None,
        "trace_url": f"{root_url}/trace" if trace_path.is_file() else None,
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
        "original": (
            "original_keyframe.jpg"
            if (_event_evidence_root(event_id) / "original_keyframe.jpg").is_file()
            else "original.jpg",
            "image/jpeg",
        ),
        "annotated": (
            "annotated_keyframe.jpg"
            if (_event_evidence_root(event_id) / "annotated_keyframe.jpg").is_file()
            else "annotated.jpg",
            "image/jpeg",
        ),
        "clip": ("evidence.mp4", "video/mp4"),
        "trace": ("trace.json", "application/json"),
    }
    if kind not in names:
        raise HTTPException(status_code=404, detail="Evidence kind not found")
    filename, media_type = names[kind]
    event_root = _event_evidence_root(event_id)
    path = (event_root / filename).resolve()
    if event_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    if kind == "clip":
        clip_path, clip_mime = _ensure_browser_evidence_clip(path)
        return FileResponse(clip_path, media_type=clip_mime, filename=clip_path.name)
    return FileResponse(path, media_type=media_type, filename=filename)


@router.patch("/events/{event_id}/review", response_model=EventResponse, tags=["events"])
def review_event(
    event_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
    user: User = Depends(reviewer_user),
):
    if payload.status == ReviewStatus.PENDING:
        raise HTTPException(status_code=422, detail="A review cannot return an event to PENDING")
    if not (event := db.get(Event, event_id)):
        raise HTTPException(status_code=404, detail="Event not found")
    if payload.status == event.review_status:
        raise HTTPException(status_code=409, detail="Duplicate review status transition")
    if payload.status == ReviewStatus.CONFIRMED:
        evidence = db.scalar(select(Evidence).where(Evidence.event_id == event.id))
        trace, _ = _read_evidence_trace(event.id)
        integrity = _evidence_integrity(event.id, evidence, trace)
        if not integrity["confirmation_ready"]:
            raise HTTPException(
                status_code=422,
                detail=(
                    "An event cannot be confirmed without complete, integrity-checked "
                    f"temporal evidence: {', '.join(integrity['confirmation_blockers'])}"
                ),
            )
    return _commit_review_transition(db, event, user, payload)


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
    v32_summary = get_v32_summary_for_stats()
    return {
        "analyzed_videos": analyzed,
        "events_by_type": by_type,
        "events_by_review_status": by_status,
        "model": detector.status(),
        "runtime": runtime,
        "event_metrics_status": "PASS_AI85_V32_EVALUATED" if v32_summary else "NOT_RUN",
        "v32_summary": v32_summary,
    }


@router.get("/benchmark/v32", tags=["benchmark"])
def benchmark_v32(_: User = Depends(current_user)):
    return get_v32_benchmark_payload()


@router.post("/benchmark/v32/seed-demo-events", tags=["benchmark"])
def benchmark_v32_seed_demo(
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    return seed_v32_demo_events(db)


@router.get("/exports/events.csv", tags=["exports"])
def export_events(db: Session = Depends(get_db), _: User = Depends(export_user)):
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
                _csv_safe(value)
                for value in (
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
                )
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
