from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ResearchReviewError(ValueError):
    pass


class ResearchReviewConflict(ResearchReviewError):
    pass


_WRITE_LOCK = threading.Lock()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchReviewError(f"Research review artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchReviewError(f"Research review artifact is invalid: {path}") from exc


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cases(source: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _read_json(source)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 13:
        raise ResearchReviewError("Research review requires the frozen 13-case proposal artifact")
    indexed = {str(row.get("case_id")): row for row in rows}
    if len(indexed) != 13 or "None" in indexed:
        raise ResearchReviewError("Research review case IDs are missing or duplicated")
    return payload, indexed


def _decision_path(decisions_dir: Path, case_id: str) -> Path:
    if not case_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in case_id
    ):
        raise ResearchReviewError("Invalid research review case ID")
    return decisions_dir / f"{case_id}.json"


def _current_decision(decisions_dir: Path, case_id: str) -> dict[str, Any] | None:
    path = _decision_path(decisions_dir, case_id)
    return _read_json(path) if path.is_file() else None


def _flatten_proposals(row: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = []
    seen = set()
    for sample in row.get("samples", []):
        for proposal in sample.get("proposals", []):
            proposal_id = str(proposal.get("proposal_id", ""))
            if not proposal_id or proposal_id in seen:
                raise ResearchReviewError(
                    f"Proposal IDs are missing or duplicated for case {row.get('case_id')}"
                )
            seen.add(proposal_id)
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "frame_index": int(sample["frame_index"]),
                    "timestamp_seconds": float(sample["timestamp_seconds"]),
                    "source": proposal["source"],
                    "confidence": float(proposal["confidence"]),
                    "geometry_context": proposal["geometry_context"],
                    "hand_proximity": float(proposal["hand_proximity"]),
                    "face_proximity": float(proposal["face_proximity"]),
                    "nearest_cross_model_iou": float(proposal["nearest_cross_model_iou"]),
                    "xyxy_in_driver_roi": [
                        float(value) for value in proposal["xyxy_in_driver_roi"]
                    ],
                }
            )
    return sorted(proposals, key=lambda item: (item["frame_index"], -item["confidence"]))


def _safe_contact_sheet(source: Path, row: dict[str, Any]) -> Path:
    reports_root = source.parent.resolve()
    path = Path(str(row["contact_sheet"]))
    candidate = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if reports_root != candidate.parent and reports_root not in candidate.parents:
        raise ResearchReviewError("Contact sheet escapes the research report directory")
    if not candidate.is_file():
        raise ResearchReviewError("Contact sheet is missing")
    return candidate


def list_phone_reviews(source: Path, decisions_dir: Path) -> dict[str, Any]:
    payload, indexed = _cases(source)
    cases = []
    for case_id, row in indexed.items():
        decision = _current_decision(decisions_dir, case_id)
        cases.append(
            {
                "case_id": case_id,
                "subject_id": row["subject_id"],
                "sample_count": len(row.get("samples", [])),
                "specialist_proposals": int(row["specialist_proposals"]),
                "auxiliary_proposals": int(row["auxiliary_proposals"]),
                "cross_model_agreement": bool(row["cross_model_agreement"]),
                "auxiliary_review_candidate": bool(row["auxiliary_review_candidate"]),
                "review_status": "REVIEWED" if decision else "PENDING",
                "decision": decision,
            }
        )
    cases.sort(key=lambda item: (item["review_status"] == "REVIEWED", item["case_id"]))
    reviewed = sum(item["review_status"] == "REVIEWED" for item in cases)
    return {
        "scope": payload["scope"],
        "cases": cases,
        "summary": {"total": len(cases), "reviewed": reviewed, "pending": len(cases) - reviewed},
    }


def phone_review_detail(source: Path, decisions_dir: Path, case_id: str) -> dict[str, Any]:
    _, indexed = _cases(source)
    if case_id not in indexed:
        raise ResearchReviewError("Research review case not found")
    row = indexed[case_id]
    contact_sheet = _safe_contact_sheet(source, row)
    return {
        "case_id": case_id,
        "subject_id": row["subject_id"],
        "sample_count": len(row.get("samples", [])),
        "specialist_proposals": int(row["specialist_proposals"]),
        "auxiliary_proposals": int(row["auxiliary_proposals"]),
        "cross_model_agreement": bool(row["cross_model_agreement"]),
        "auxiliary_review_candidate": bool(row["auxiliary_review_candidate"]),
        "contact_sheet_sha256": _file_sha256(contact_sheet),
        "proposals": _flatten_proposals(row),
        "decision": _current_decision(decisions_dir, case_id),
    }


def phone_review_contact_sheet(source: Path, case_id: str) -> Path:
    _, indexed = _cases(source)
    if case_id not in indexed:
        raise ResearchReviewError("Research review case not found")
    return _safe_contact_sheet(source, indexed[case_id])


def save_phone_review(
    source: Path,
    decisions_dir: Path,
    case_id: str,
    decision: dict[str, Any],
    reviewer_id: str,
) -> dict[str, Any]:
    payload, indexed = _cases(source)
    if case_id not in indexed:
        raise ResearchReviewError("Research review case not found")
    row = indexed[case_id]
    proposals = {item["proposal_id"]: item for item in _flatten_proposals(row)}
    selected_id = decision["selected_proposal_id"]
    if selected_id != "NO_VALID_PROPOSAL" and selected_id not in proposals:
        raise ResearchReviewError("Selected proposal does not belong to this case")

    with _WRITE_LOCK:
        current = _current_decision(decisions_dir, case_id)
        current_revision = current.get("revision") if current else None
        if decision.get("expected_revision") != current_revision:
            raise ResearchReviewConflict("Review changed in another session; reload before saving")
        contact_sheet = _safe_contact_sheet(source, row)
        receipt = {
            "schema_version": 1,
            "case_id": case_id,
            "reviewer_id": reviewer_id,
            "reviewer_type": "HUMAN",
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
            "phone_visibility": decision["phone_visibility"],
            "phone_use_assessment": decision["phone_use_assessment"],
            "selected_proposal_id": selected_id,
            "selected_proposal": proposals.get(selected_id),
            "notes": decision["notes"].strip(),
            "source": {
                "proposal_artifact": source.as_posix(),
                "proposal_artifact_sha256": _file_sha256(source),
                "source_case_sha256": _canonical_sha256(row),
                "contact_sheet_sha256": _file_sha256(contact_sheet),
                "experiment_scope": payload["scope"],
            },
            "previous_revision": current_revision,
            "training_admission": "NOT_AUTOMATIC_REQUIRES_GOVERNED_MATERIALIZATION",
            "production_approval": False,
        }
        receipt["revision"] = _canonical_sha256(receipt)
        latest = _decision_path(decisions_dir, case_id)
        history = decisions_dir / "history" / case_id
        decisions_dir.mkdir(parents=True, exist_ok=True)
        history.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
        history_path = (
            history / f"{receipt['reviewed_at_utc'].replace(':', '-')}_{receipt['revision']}.json"
        )
        history_path.write_text(encoded, encoding="utf-8")
        temporary = latest.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(latest)
    return receipt
