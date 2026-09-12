"""Human-operated final payload/evidence collection; no canonical or governance promotion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from training.validate_review1 import digest, payload_digest

CLIPS = ("C01", "C07", "C10", "C11")
KINDS = ("payload", "rights", "lineage")
HUMAN = ("reviewer_id", "reviewed_at", "decision", "review_notes")
MAX_UPLOAD = 20 * 1024 * 1024
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".json", ".csv", ".html"}


def reject(message, code=400):
    raise HTTPException(code, message)


def read_json(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except (OSError, ValueError):
        reject("Thiếu hoặc lỗi tệp hồ sơ local. Kiểm tra gói bản tổng hợp.", 409)


def within(root, value):
    if not isinstance(value, str) or not value:
        reject("Đường dẫn bằng chứng bị thiếu.", 409)
    p = (root / value).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        reject("Tệp được tham chiếu không tồn tại trong dự án.", 409)
    return p


def require_text(body, name, label, minimum=1):
    value = body.get(name)
    if not isinstance(value, str) or len(value.strip()) < minimum:
        reject(f"{label}: cần ít nhất {minimum} ký tự.")
    return value.strip()


def human_input(body):
    reviewer = require_text(body, "reviewer_id", "Danh tính người duyệt", 3)
    words = set(re.findall(r"[a-z]+", reviewer.lower()))
    if (
        not words
        or words
        & {
            "ai",
            "codex",
            "bot",
            "test",
            "dummy",
            "example",
            "human",
            "reviewer",
            "placeholder",
            "unknown",
            "pending",
            "none",
            "null",
            "tbd",
            "assistant",
        }
        or any(c in reviewer for c in "<>")
    ):
        reject("Nhập mã hoặc tên người duyệt thật, không dùng danh tính mẫu.")
    stamp = require_text(body, "reviewed_at", "Thời điểm xác nhận")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError()
    except ValueError:
        reject("Thời điểm phải có múi giờ. Có thể bấm Điền giờ hiện tại.")
    notes = require_text(body, "review_notes", "Ghi chú xác nhận", 10)
    if body.get("attested") is not True:
        reject("Bạn cần tích xác nhận chịu trách nhiệm về thông tin vừa cung cấp.")
    return {"reviewer_id": reviewer, "reviewed_at": stamp, "review_notes": notes}


def atomic_save(target, value, expected, archive):
    """Archive previous/current bytes, then replace a single current document."""
    actual = digest(target) if target.exists() else None
    if actual != expected:
        reject("Hồ sơ vừa thay đổi. Tải lại trang trước khi gửi.", 409)
    folder = archive / uuid.uuid4().hex
    folder.mkdir(parents=True)
    if target.exists():
        (folder / "before.json").write_bytes(target.read_bytes())
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    (folder / "submitted.json").write_bytes(raw)
    temp = folder / "replace.tmp"
    temp.write_bytes(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (digest(target) if target.exists() else None) != expected:
        reject("Hồ sơ bị sửa bên ngoài trong lúc lưu; chưa áp dụng lần gửi này.", 409)
    os.replace(temp, target)
    return hashlib.sha256(raw).hexdigest()


def attach_final_review(app, root, queue_path, lock, token, static):
    base = queue_path.parent.parent
    candidates = base / "final_payload_candidates"
    reviews = base / "final_payload_review"
    submissions = base / "human_review_return/governance_ui"
    uploads = submissions / "uploads"

    def paths(clip):
        if clip not in CLIPS:
            reject("Clip không thuộc gói bốn video cần xác nhận.", 404)
        return (
            candidates / f"{clip}_final_payload_candidate.json",
            reviews / f"{clip}_final_payload_receipt.json",
        )

    def target_for(clip, kind):
        cp, rp = paths(clip)
        if kind not in KINDS:
            reject("Loại xác nhận không hợp lệ.", 404)
        return rp if kind == "payload" else submissions / f"{clip}_{kind}.json"

    def load_candidate(clip, deep=False):
        cp, rp = paths(clip)
        c, template = read_json(cp), read_json(rp)
        try:
            assert c["clip_id"] == template["clip_id"] == clip
            assert c["dataset_role"] == "TEMPORAL_DEVELOPMENT"
            assert template["candidate_sha256"] == digest(cp)
            assert within(root, template["candidate_path"]) == cp.resolve()
            payload = c["assembled_payload"]
            assert set(payload) == {"rights", "physical_lineage", "identity", "sequence"}
            assert payload_digest(payload) == c["assembled_payload_sha256"]
            assert c["assembled_payload_sha256"] == template["final_payload_sha256"]
            assert c["video_sha256"] == template["video_sha256"]
            assert template["record_type"] == "sequence"
            assert template["payload_sha256"] == payload_digest(payload["sequence"])
            for kind, part in payload.items():
                assert payload_digest(part) == c["assembled_record_payload_sha256"][kind]
                assert payload_digest(part) == template["record_payload_sha256"][kind]
            if deep:
                assert digest(within(root, c["video_path"])) == c["video_sha256"]
                assert within(root, c["source_queue"]["path"]) == queue_path.resolve()
                assert digest(queue_path) == c["source_queue"]["sha256"]
                with queue_path.open(encoding="utf-8-sig", newline="") as f:
                    rows = [r for r in csv.DictReader(f) if r["clip_id"] == clip]
                row_map = {r["item_id"]: r for r in rows}
                bindings = c["approved_item_receipts"]
                assert len(row_map) == len(rows) == len(bindings)
                assert set(row_map) == {b["item_id"] for b in bindings}
                for kind, source in c["source_review1_records"].items():
                    sp = within(root, source["path"])
                    assert digest(sp) == source["sha256"]
                    assert source["sha256"] == c["source_review1_record_sha256"][source["path"]]
                    rec = read_json(sp)
                    assert rec["status"] == "REVIEW1_PROPOSAL" and rec["human_approved"] is False
                    assert rec["video_sha256"] == c["video_sha256"]
                    assert rec["payload"] == payload[kind]
                    assert payload_digest(rec["payload"]) == source["payload_sha256"]
                pointers = set()
                for b in bindings:
                    row = row_map[b["item_id"]]
                    ep = within(root, b["path"])
                    assert digest(ep) == b["sha256"] == row["REVIEW_EVIDENCE_SHA256"]
                    assert ep == within(root, row["REVIEW_EVIDENCE_PATH"])
                    evidence = read_json(ep)
                    for k, column in {
                        "item_id": "item_id",
                        "video_sha256": "video_sha256",
                        "record_sha256": "record_sha256",
                        "payload_sha256": "payload_sha256",
                        "reviewer_id": "REVIEWER_ID",
                        "reviewed_at": "REVIEWED_AT",
                        "decision": "HUMAN_DECISION",
                        "review_notes": "HUMAN_NOTES",
                    }.items():
                        assert evidence[k] == row[column]
                    assert evidence["decision"] == b["decision"] == "APPROVE"
                    assert row["video_sha256"] == c["video_sha256"]
                    assert row["record_sha256"] == b["source_record_sha256"]
                    assert row["payload_sha256"] == b["source_payload_sha256"]
                    assert row["record_path"] == b["source_record_path"]
                    assert row["REVIEWER_ID"] == b["reviewer_id"]
                    assert row["REVIEWED_AT"] == b["reviewed_at"]
                    assert not row["EDITED_ITEM_JSON"]
                    pointer = b["assembled_item_pointer"]
                    assert pointer not in pointers
                    pointers.add(pointer)
                    item = payload
                    for segment in pointer.strip("/").split("/"):
                        item = item[int(segment)] if isinstance(item, list) else item[segment]
                    assert item == b["approved_item"] == evidence["item"]
                    assert item == json.loads(row["proposed_item_json"])
                    assert payload_digest(item) == b["approved_item_sha256"]
                expected_pointers = {"/rights", "/physical_lineage", "/identity"}
                for key in ("phone_intervals", "seatbelt_intervals", "context_intervals"):
                    expected_pointers.update(
                        f"/sequence/{key}/{i}" for i in range(len(payload["sequence"][key]))
                    )
                assert pointers == expected_pointers
        except (AssertionError, KeyError, TypeError, ValueError, IndexError, OSError):
            reject(
                "Bản tổng hợp, video hoặc biên nhận nguồn không còn khớp. "
                "Dừng xác nhận và kiểm tra SHA.",
                409,
            )
        return c, template, digest(cp)

    def documents(ids, clip, kind):
        if not isinstance(ids, list) or len(ids) > 10 or any(not isinstance(i, str) for i in ids):
            reject("Danh sách evidence không hợp lệ; tối đa 10 tệp mỗi lần gửi.")
        if len(set(ids)) != len(ids):
            reject("Một tệp evidence bị chọn hai lần.")
        result = []
        for uid in ids:
            if not re.fullmatch(r"[0-9a-f]{32}", uid):
                reject("Mã tệp evidence không hợp lệ.")
            info = read_json(uploads / uid / "metadata.json")
            p = uploads / uid / "content"
            if info.get("clip_id") != clip or info.get("kind") != kind:
                reject("Tệp evidence thuộc clip hoặc loại duyệt khác.")
            if not p.is_file() or not p.stat().st_size or digest(p) != info.get("sha256"):
                reject("Tệp evidence đã thay đổi hoặc rỗng. Chọn lại tệp đúng.", 409)
            result.append({**info, "path": p.relative_to(root).as_posix()})
        return result

    @app.get("/final")
    def final_page():
        return FileResponse(static / "final.html")

    @app.get("/api/final-review")
    def overview():
        with lock:
            result = []
            for clip in CLIPS:
                try:
                    c, receipt, version = load_candidate(clip)
                    states = {}
                    for kind in KINDS:
                        target = target_for(clip, kind)
                        value = read_json(target) if target.exists() else None
                        states[kind] = {
                            "version": digest(target) if target.exists() else None,
                            "submitted": bool(value and value.get("decision")),
                            "document": value if value and value.get("decision") else None,
                        }
                    result.append(
                        {
                            "clip_id": clip,
                            "candidate": c,
                            "version": version,
                            "states": states,
                            "error": None,
                        }
                    )
                except HTTPException as e:
                    result.append({"clip_id": clip, "error": e.detail})
            return {"clips": result, "token": token, "governance_promotion": False}

    @app.get("/api/final-review/{clip}/download/{kind}")
    def download(clip: str, kind: str):
        target = paths(clip)[0] if kind == "candidate" else target_for(clip, kind)
        if not target.is_file():
            reject("Chưa có hồ sơ để tải.", 404)
        return FileResponse(target, media_type="application/json", filename=target.name)

    @app.post("/api/final-review/{clip}/evidence/{kind}")
    async def upload(clip: str, kind: str, request: Request):
        paths(clip)
        if kind not in ("rights", "lineage"):
            reject("Chỉ tải evidence quyền sử dụng hoặc nguồn gốc.")
        name = unquote(request.headers.get("x-file-name", ""))
        if (
            not name
            or len(name) > 180
            or any(c in name for c in "/\\\r\n\x00")
            or Path(name).suffix.lower() not in EXTENSIONS
        ):
            reject("Chọn PDF, ảnh, văn bản, JSON, CSV hoặc HTML; tên tệp không có đường dẫn.")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                reject("Mỗi tệp evidence tối đa 20 MB. Chọn tệp nhỏ hơn.", 413)
            chunks.append(chunk)
        if not size:
            reject("Tệp evidence rỗng. Chọn tệp có nội dung.")
        uid = uuid.uuid4().hex
        folder = uploads / uid
        folder.mkdir(parents=True)
        raw = b"".join(chunks)
        (folder / "content").write_bytes(raw)
        info = {
            "upload_id": uid,
            "clip_id": clip,
            "kind": kind,
            "filename": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": size,
        }
        (folder / "metadata.json").write_text(
            json.dumps(info, ensure_ascii=False), encoding="utf-8"
        )
        return info

    @app.get("/api/final-review/evidence/{uid}")
    def download_evidence(uid: str):
        if not re.fullmatch(r"[0-9a-f]{32}", uid):
            reject("Không tìm thấy tệp evidence.", 404)
        info = read_json(uploads / uid / "metadata.json")
        return FileResponse(
            uploads / uid / "content",
            media_type="application/octet-stream",
            filename=info["filename"],
        )

    @app.post("/api/final-review/{clip}/{kind}")
    def submit(clip: str, kind: str, body: dict):
        with lock:
            target = target_for(clip, kind)
            person = human_input(body)
            c, template, version = load_candidate(clip, deep=True)
            if body.get("candidate_version") != version:
                reject("Bản tổng hợp đã đổi. Tải lại và xác nhận đúng bản đang xem.", 409)
            current_version = digest(target) if target.exists() else None
            if body.get("document_version") != current_version:
                reject("Có hồ sơ mới hơn. Tải lại trang để không ghi đè dữ liệu.", 409)
            decision = body.get("decision")
            if kind == "payload":
                if any(template.get(k) for k in HUMAN):
                    reject("Receipt cuối đã có dữ liệu người duyệt. Không ghi đè.", 409)
                if decision != "APPROVE":
                    reject(
                        "Chỉ xác nhận nguyên trạng trên trang này. "
                        "Bản cần sửa phải được chuẩn bị riêng."
                    )
                result = {
                    **template,
                    **person,
                    "decision": "APPROVE",
                    "template_status": "HUMAN_CONFIRMATION_RECORDED",
                    "receipt_type": "HUMAN_UI_FULL_PAYLOAD_CONFIRMATION",
                    "reviewer_type": "HUMAN",
                    "reviewer_origin": "HUMAN",
                    "reviewer_agent": None,
                    "submitted_via": "LOCAL_FINAL_REVIEW_UI",
                    "governance_promotion": False,
                }
            else:
                evidence = documents(body.get("upload_ids", []), clip, kind)
                if evidence and body.get("evidence_confirmed") is not True:
                    reject("Xác nhận đã đọc các tệp evidence và đối chiếu với video này.")
                result = {
                    "clip_id": clip,
                    "video_sha256": c["video_sha256"],
                    "candidate_path": paths(clip)[0].relative_to(root).as_posix(),
                    "candidate_sha256": version,
                    "final_payload_sha256": c["assembled_payload_sha256"],
                    **person,
                    "reviewer_type": "HUMAN",
                    "reviewer_origin": "HUMAN",
                    "reviewer_agent": None,
                    "decision": decision,
                    "evidence": evidence,
                    "submission_status": "PENDING_EVIDENCE_VERIFICATION",
                    "governance_promotion": False,
                    "canonical_conversion": False,
                }
                if kind == "rights":
                    if decision not in ("ACCEPT", "REJECT", "NEEDS_HUMAN_DECISION"):
                        reject("Chọn quyết định quyền sử dụng.")
                    if decision in ("ACCEPT", "REJECT") and not evidence:
                        reject("Cần tải bằng chứng thực làm căn cứ cho quyết định quyền sử dụng.")
                    details = {}
                    for key, label in (
                        ("source_page_url", "Trang nguồn"),
                        ("license_or_terms_url", "Trang giấy phép/điều khoản"),
                        ("project_use", "Mục đích sử dụng"),
                    ):
                        value = require_text(body, key, label)
                        if key.endswith("url") and (
                            urlparse(value).scheme not in ("http", "https")
                            or not urlparse(value).netloc
                        ):
                            reject(f"{label}: cần URL http hoặc https đầy đủ.")
                        details[key] = value
                    for key in ("creator", "asset_id", "downloaded_at", "download_method"):
                        value = body.get(key, "")
                        if not isinstance(value, str):
                            reject("Thông tin nguồn phải là văn bản.")
                        details[key] = value.strip() or None
                    result.update(
                        details,
                        project_use_review={
                            "ACCEPT": "HUMAN_APPROVED",
                            "REJECT": "REJECTED",
                            "NEEDS_HUMAN_DECISION": "PENDING",
                        }[decision],
                    )
                else:
                    if decision not in ("SUBMIT_FOR_VERIFICATION", "NOT_PROVABLE"):
                        reject("Chọn Cung cấp bằng chứng hoặc Chưa chứng minh được.")
                    ids = body.get("physical_ids", {})
                    if not isinstance(ids, dict):
                        reject("Thông tin định danh không hợp lệ.")
                    fields = (
                        "source_id",
                        "camera_id",
                        "capture_session_id",
                        "physical_vehicle_group_id",
                    )
                    if decision == "SUBMIT_FOR_VERIFICATION":
                        if not evidence:
                            reject("Cần bằng chứng thực cho các định danh vật lý.")
                        values = {k: require_text(ids, k, k) for k in fields}
                        for v in values.values():
                            if v.lower() in {"unknown", "pending", "tbd", "null", "none", "n/a"}:
                                reject("Không dùng giá trị chưa biết để chứng minh định danh.")
                        groups = ids.get("person_group_ids")
                        if (
                            not isinstance(groups, list)
                            or not groups
                            or any(
                                not isinstance(g, str)
                                or not g.strip()
                                or g.strip().lower()
                                in {"unknown", "pending", "tbd", "null", "none", "n/a"}
                                for g in groups
                            )
                            or len(set(groups)) != len(groups)
                        ):
                            reject("Cần danh sách mã nhóm người thật, không trống hoặc trùng nhau.")
                        mapping = ids.get("vehicle_physical_groups", {})
                        if not isinstance(mapping, dict) or any(
                            not k.strip() or not isinstance(v, str) or not v.strip()
                            for k, v in mapping.items()
                        ):
                            reject("Bảng xe–nhóm vật lý không hợp lệ.")
                        if mapping and values["physical_vehicle_group_id"] not in mapping.values():
                            reject("Nhóm xe chính phải nằm trong bảng ánh xạ xe.")
                        if body.get("multiple_vehicles") not in (True, False):
                            reject("Xác nhận clip có nhiều xe hay không.")
                        if body["multiple_vehicles"] and len(mapping) < 2:
                            reject("Clip nhiều xe cần ánh xạ đầy đủ ít nhất hai xe.")
                        result.update(
                            physical_ids={
                                **values,
                                "person_group_ids": groups,
                                "vehicle_physical_groups": mapping,
                            },
                            multiple_vehicles=body["multiple_vehicles"],
                            physical_lineage_status="PENDING_VERIFICATION",
                        )
                    else:
                        if any(v not in (None, "", [], {}) for v in ids.values()):
                            reject("Khi chưa chứng minh được, để trống các mã định danh.")
                        result.update(physical_ids=None, physical_lineage_status="NOT_PROVABLE")
            saved_sha = atomic_save(target, result, current_version, submissions / "history")
            return {
                "saved": True,
                "path": target.relative_to(root).as_posix(),
                "sha256": saved_sha,
                "governance_promotion": False,
                "message": "Đã lưu xác nhận của bạn. Các điều kiện dữ liệu vẫn cần kiểm tra riêng.",
            }
