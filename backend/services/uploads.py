from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_MIME_PREFIXES = ("video/", "application/octet-stream")


def safe_upload_name(original: str) -> str:
    base = Path(original).name
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(base).stem)[:80] or "video"
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported video extension")
    return f"{stem}{suffix}"


async def save_upload(upload: UploadFile, destination: Path, maximum_bytes: int) -> tuple[int, str]:
    if upload.content_type:
        media_type = upload.content_type.split(";", 1)[0].strip().lower()
        if not (media_type.startswith("video/") or media_type == "application/octet-stream"):
            raise HTTPException(status_code=415, detail="Unsupported media type")
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    digest = hashlib.sha256()
    try:
        with destination.open("xb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > maximum_bytes:
                    raise HTTPException(status_code=413, detail="Upload exceeds configured maximum")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()

