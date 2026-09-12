"""Local human-operated Review1 queue UI. Never creates canonical truth or promotes gates."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import secrets
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tools.temporal_reviewer.final_review import attach_final_review
from training.validate_review1 import digest, payload_digest, validate_record

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent
QUEUE = ROOT / "datasets/incoming/v2_sequence_001/review1/HUMAN_APPROVAL_QUEUE.csv"
HUMAN_FIELDS = (
    "HUMAN_DECISION",
    "REVIEWER_ID",
    "REVIEWED_AT",
    "HUMAN_NOTES",
    "EDITED_ITEM_JSON",
    "REVIEW_EVIDENCE_PATH",
    "REVIEW_EVIDENCE_SHA256",
)
DECISIONS = {"APPROVE", "EDIT_AND_APPROVE", "REJECT", "UNCERTAIN"}


def fail(message, status=400):
    raise HTTPException(status, message)


def read_queue(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    fields = reader.fieldnames or []
    if not set(HUMAN_FIELDS).issubset(fields) or not rows:
        fail("Hàng đợi trống hoặc thiếu cột người duyệt.")
    if len({r["item_id"] for r in rows}) != len(rows):
        fail("Hàng đợi có mã mục trùng nhau.")
    return rows, fields, hashlib.sha256(raw).hexdigest(), raw


def pending(row):
    return all(not row.get(k, "").strip() for k in HUMAN_FIELDS)


def resolve_local(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        fail("Không tìm thấy tệp hợp lệ trong thư mục dự án.")
    return path


def replace_item(record, row, value):
    original = json.loads(row["proposed_item_json"])
    payload = record["payload"]
    if row["record_type"] != "sequence":
        if payload != original:
            fail("Đề xuất trong hàng đợi không khớp hồ sơ nguồn.", 409)
        record["payload"] = value
        return
    matches = [
        (key, i)
        for key in ("phone_intervals", "seatbelt_intervals", "context_intervals")
        for i, item in enumerate(payload.get(key, []))
        if item == original
    ]
    if len(matches) != 1:
        fail("Khoảng thời gian không khớp duy nhất với đề xuất nguồn.", 409)
    key, index = matches[0]
    payload[key][index] = value


def create_app(queue_path=QUEUE, root=ROOT):
    app = FastAPI(title="Duyệt video temporal", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    queue_path, root = Path(queue_path), Path(root).resolve()

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if (
                origin and origin != str(request.base_url).rstrip("/")
            ) or not secrets.compare_digest(request.headers.get("x-review-token", ""), token):
                return JSONResponse({"detail": "Phiên duyệt không hợp lệ. Hãy tải lại trang."}, 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "media-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name in {"app.js", "style.css", "final.js", "final.css"}:
            media_type = "text/javascript" if name.endswith(".js") else "text/css"
            return FileResponse(STATIC / name, media_type=media_type)
        if name == "heading.woff2":
            return FileResponse(ROOT / "frontend/src/assets/IBMPlexSansCondensed-SemiBold.woff2")
        fail("Không tìm thấy tài nguyên.", 404)

    @app.get("/api/queue")
    def queue():
        try:
            with lock:
                rows, _, version, _ = read_queue(queue_path)
            return {
                "items": rows,
                "version": version,
                "token": token,
                "governance_promotion": False,
            }
        except OSError:
            fail("Chưa có hàng đợi local. Kiểm tra HUMAN_APPROVAL_QUEUE.csv.", 404)

    @app.get("/api/video/{clip_id}")
    def video(clip_id: str):
        rows, _, _, _ = read_queue(queue_path)
        paths = {r["video_path"] for r in rows if r["clip_id"] == clip_id}
        if len(paths) != 1:
            fail("Không tìm thấy video duy nhất cho clip.", 404)
        return FileResponse(resolve_local(root, paths.pop()), media_type="video/mp4")

    @app.get("/api/export")
    def export():
        return FileResponse(queue_path, filename="HUMAN_APPROVAL_QUEUE.csv", media_type="text/csv")

    @app.post("/api/decisions")
    def decide(body: dict):
        with lock:
            rows, fields, version, raw = read_queue(queue_path)
            if body.get("version") != version:
                fail("Hàng đợi đã thay đổi. Tải lại trước khi xác nhận.", 409)
            if not isinstance(body.get("reviewer_id"), str):
                fail("Nhập danh tính người duyệt thật.")
            reviewer = body["reviewer_id"].strip()
            words = set(re.findall(r"[a-z]+", reviewer.lower()))
            forbidden = {
                "ai",
                "codex",
                "bot",
                "test",
                "dummy",
                "example",
                "unknown",
                "human",
                "reviewer",
                "placeholder",
                "pending",
                "tbd",
                "assistant",
                "none",
                "null",
            }
            if (
                len(reviewer) < 3
                or not words
                or words & forbidden
                or any(c in reviewer for c in "<>")
            ):
                fail("Nhập danh tính người duyệt thật; không dùng AI hoặc tên mẫu.")
            reviewed_at = body.get("reviewed_at", "")
            try:
                stamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
                if stamp.tzinfo is None or stamp.utcoffset() is None:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                fail("Thời điểm duyệt phải theo ISO-8601 và có múi giờ.")
            notes = str(body.get("notes", "")).strip()
            if len(notes) < 10 or body.get("attested") is not True:
                fail("Cần ghi chú duyệt và xác nhận đã trực tiếp xem nội dung liên quan.")
            changes = body.get("items")
            if not isinstance(changes, list) or not 1 <= len(changes) <= len(rows):
                fail("Chọn ít nhất một mục để ghi quyết định.")
            if any(not isinstance(c, dict) for c in changes):
                fail("Danh sách quyết định không hợp lệ.")
            ids = [c.get("item_id") for c in changes]
            if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
                fail("Mã mục không hợp lệ hoặc bị lặp.")
            lookup = {r["item_id"]: r for r in rows}
            records, candidates, hashes = {}, {}, {}
            for change in changes:
                row = lookup.get(change["item_id"])
                if not row or not pending(row):
                    fail("Mục không tồn tại hoặc đã có dữ liệu người duyệt; không ghi đè.", 409)
                if change.get("decision") not in DECISIONS:
                    fail("Quyết định không hợp lệ.")
                record_path = resolve_local(root, row["record_path"])
                video_path = resolve_local(root, row["video_path"])
                for path, expected in (
                    (record_path, row["record_sha256"]),
                    (video_path, row["video_sha256"]),
                ):
                    if path not in hashes:
                        hashes[path] = digest(path)
                    if hashes[path] != expected:
                        fail("Video hoặc đề xuất đã đổi SHA. Cần duyệt lại bản hiện tại.", 409)
                if row["record_path"] not in records:
                    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
                    records[row["record_path"]] = record
                    candidates[row["record_path"]] = copy.deepcopy(record)
                record = records[row["record_path"]]
                if (
                    payload_digest(record["payload"]) != row["payload_sha256"]
                    or record["video_sha256"] != row["video_sha256"]
                    or record["record_type"] != row["record_type"]
                    or resolve_local(root, record["video_path"]) != video_path
                ):
                    fail("Liên kết nội dung đề xuất không khớp.", 409)
                proposed = json.loads(row["proposed_item_json"])
                replace_item(copy.deepcopy(record), row, proposed)
                if change["decision"] == "EDIT_AND_APPROVE":
                    if not isinstance(change.get("edited_item"), dict):
                        fail("Cần toàn bộ nội dung JSON đã sửa.")
                    replace_item(candidates[row["record_path"]], row, change["edited_item"])
                elif change.get("edited_item") is not None:
                    fail("Chỉ quyết định Sửa và phê duyệt mới được kèm nội dung sửa.")
            # Include earlier human edits when checking full-timeline consistency.
            for row in rows:
                if row["record_path"] in candidates and row["HUMAN_DECISION"] == "EDIT_AND_APPROVE":
                    replace_item(
                        candidates[row["record_path"]], row, json.loads(row["EDITED_ITEM_JSON"])
                    )
            for record in candidates.values():
                record["video_path"] = str(resolve_local(root, record["video_path"]))
                errors = validate_record(record)
                if errors:
                    fail("Nội dung không đạt hợp đồng Review1: " + "; ".join(errors[:5]))
            batch_id = uuid.uuid4().hex
            folder = queue_path.parent / "human_decisions" / batch_id
            folder.mkdir(parents=True, exist_ok=False)
            # Durable original snapshot before the single atomic queue replacement.
            (folder / "queue_before.csv").write_bytes(raw)
            for change in changes:
                row = lookup[change["item_id"]]
                receipt = {
                    "receipt_type": "HUMAN_UI_ITEM_DECISION",
                    "batch_id": batch_id,
                    "item_id": row["item_id"],
                    "item": json.loads(row["proposed_item_json"]),
                    "video_sha256": row["video_sha256"],
                    "record_sha256": row["record_sha256"],
                    "payload_sha256": row["payload_sha256"],
                    "record_type": row["record_type"],
                    "reviewer_id": reviewer,
                    "reviewed_at": reviewed_at,
                    "decision": change["decision"],
                    "review_notes": notes,
                    "edited_item": change.get("edited_item"),
                    "edited_item_sha256": payload_digest(change["edited_item"])
                    if change.get("edited_item") is not None
                    else None,
                    "attestation": "Human explicitly confirmed viewing and submitting these items.",
                    "governance_promotion": False,
                    "canonical_conversion": False,
                }
                receipt_path = folder / f"receipt_{ids.index(row['item_id']):03d}.json"
                receipt_path.write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                row.update(
                    HUMAN_DECISION=change["decision"],
                    REVIEWER_ID=reviewer,
                    REVIEWED_AT=reviewed_at,
                    HUMAN_NOTES=notes,
                    EDITED_ITEM_JSON=json.dumps(change["edited_item"], ensure_ascii=False)
                    if change.get("edited_item") is not None
                    else "",
                    REVIEW_EVIDENCE_PATH=receipt_path.relative_to(root).as_posix(),
                    REVIEW_EVIDENCE_SHA256=digest(receipt_path),
                )
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            staged = folder / "queue_after.csv"
            staged.write_text(stream.getvalue(), encoding="utf-8", newline="")
            temp = folder / "queue_replace.tmp"
            temp.write_bytes(staged.read_bytes())
            if digest(queue_path) != version:
                fail("Hàng đợi vừa được sửa bên ngoài. Chưa áp dụng quyết định.", 409)
            os.replace(temp, queue_path)
            return {"saved": len(changes), "batch_id": batch_id, "governance_promotion": False}

    attach_final_review(app, root, queue_path, lock, token, STATIC)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8766)
