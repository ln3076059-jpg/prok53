from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from training.common import YoloLabel, sha256_file, stable_json_hash
from training.run_review1_review import REVIEW1_STATUSES, REVIEWER_ID, VISUAL_METHODS

app = FastAPI(title="Roadwatch Annotation Reviewer")
manifest_path = Path("datasets/manifests/review_queue.json")
decisions_path = Path("datasets/manifests/review_decisions.jsonl")
acknowledgements_path = Path("datasets/manifests/model_proposal_batch_acknowledgements.jsonl")
pending_only = False
decision_lock = Lock()
datasets_root = Path("datasets")


class Annotation(BaseModel):
    box_id: str | None = Field(default=None, max_length=256)
    class_id: int = Field(ge=0, le=2)
    yolo: tuple[float, float, float, float]
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|PENDING|UNCERTAIN)$",
    )

    def validate(self) -> None:
        YoloLabel(self.class_id, *self.yolo).validate()


class Decision(BaseModel):
    sample_id: str = Field(min_length=1, max_length=256)
    reviewer_id: str = Field(min_length=2, max_length=256)
    reviewer_type: str = Field(pattern="^(HUMAN|AI)$")
    status: str = Field(
        pattern="^(APPROVED|APPROVED_NEGATIVE|REJECTED|UNCERTAIN|"
        + "|".join(sorted(REVIEW1_STATUSES))
        + ")$"
    )
    decision_reason: str = Field(
        min_length=3,
        max_length=128,
        pattern=(
            "^(PHYSICAL_PHONE_CONFIRMED|PHONE_FALSE_POSITIVE|PHONE_MISSING_LABEL|"
            "PHONE_AMBIGUOUS|MOUNTED_OR_STATIC_PHONE|UPPER_BODY_ROI_CONFIRMED|"
            "SEATBELT_STATE_CONFIRMED|UNCERTAIN_OR_OCCLUDED|HARD_NEGATIVE_CONFIRMED|"
            "INSUFFICIENT_EVIDENCE|OTHER_DOCUMENTED_IN_NOTES)$"
        ),
    )
    notes: str = Field(default="", max_length=4000)
    annotations: list[Annotation]
    vehicle_context_id: str | None = Field(default=None, max_length=256)
    video_id: str | None = Field(default=None, max_length=256)
    vehicle_id: str | None = Field(default=None, max_length=256)
    person_id: str | None = Field(default=None, max_length=256)
    camera_id: str | None = Field(default=None, max_length=256)
    conditions: list[str] = Field(default_factory=list, max_length=16)
    delegated_by: str | None = Field(default=None, max_length=256)
    approval_authority_id: str | None = Field(default=None, max_length=256)
    review1_confidence: float | None = Field(default=None, ge=0, le=1)
    risk_flags: list[str] = Field(default_factory=list, max_length=32)
    visual_evidence: dict = Field(default_factory=dict)
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(PENDING|driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|UNCERTAIN)$",
    )
    def model_post_init(self, __context) -> None:
        if self.reviewer_type == "AI":
            if self.reviewer_id != REVIEWER_ID:
                raise ValueError(f"AI decisions require reviewer_id={REVIEWER_ID}")
            if self.status not in REVIEW1_STATUSES:
                raise ValueError("AI reviewers cannot emit human decision statuses")
            if self.delegated_by != "admin" or self.approval_authority_id != "admin":
                raise ValueError("review1 decisions require explicit admin delegation provenance")
            if self.review1_confidence is None:
                raise ValueError("review1 decisions require review1_confidence")
            if (
                self.visual_evidence.get("inspected") is not True
                or self.visual_evidence.get("method") not in VISUAL_METHODS
            ):
                raise ValueError("review1 decisions require explicit visual inspection evidence")
            if self.status == "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION" and (
                self.review1_confidence < 0.97 or self.risk_flags
            ):
                raise ValueError("Review 1 Tier A requires confidence >= 0.97 and no risks")
            if self.status == "REVIEW1_ACCEPTED_PROPOSAL" and (
                not 0.85 <= self.review1_confidence < 0.97 or self.risk_flags
            ):
                raise ValueError("Review 1 Tier B requires 0.85 <= confidence < 0.97 and no risks")
            if (
                self.decision_reason == "PHONE_MISSING_LABEL"
                and self.status != "REVIEW1_CORRECTION_PROPOSAL"
            ):
                raise ValueError("PHONE_MISSING_LABEL requires a Review 1 correction proposal")
            if self.decision_reason == "UNCERTAIN_OR_OCCLUDED" and any(
                annotation.class_id == 2 for annotation in self.annotations
            ):
                raise ValueError("uncertain belt evidence cannot create an unfastened annotation")
        elif self.status in REVIEW1_STATUSES:
            raise ValueError("REVIEW1 statuses require reviewer_type=AI")
        elif self.reviewer_id == REVIEWER_ID:
            raise ValueError("review1 identity cannot claim reviewer_type=HUMAN")
        if self.decision_reason == "OTHER_DOCUMENTED_IN_NOTES" and not self.notes.strip():
            raise ValueError("OTHER_DOCUMENTED_IN_NOTES requires review notes")
        approved = self.status in {"APPROVED", "APPROVED_NEGATIVE"}
        if approved and not (self.vehicle_context_id or "").strip():
            raise ValueError("approved samples require a vehicle_context_id")
        if approved and self.occupant_role in {"PENDING", "UNCERTAIN"}:
            raise ValueError("approved samples require a resolved occupant_role")
        if approved:
            required_metadata = {
                "video_id": self.video_id,
                "vehicle_id": self.vehicle_id,
                "person_id": self.person_id,
                "camera_id": self.camera_id,
            }
            missing = [
                name for name, value in required_metadata.items() if not (value or "").strip()
            ]
            if missing:
                raise ValueError(f"approved samples require metadata: {', '.join(missing)}")
            if not self.conditions:
                raise ValueError("approved samples require at least one confirmed condition")
        if self.status == "APPROVED" and any(
            annotation.occupant_role in {"PENDING", "UNCERTAIN"} for annotation in self.annotations
        ):
            raise ValueError("APPROVED annotations require a resolved occupant_role per box")
        if self.status == "APPROVED" and not self.annotations:
            raise ValueError("APPROVED samples require at least one annotation")
        if self.status == "APPROVED_NEGATIVE" and self.annotations:
            raise ValueError("APPROVED_NEGATIVE samples must not contain annotations")


class BatchAcknowledgement(BaseModel):
    reviewer_id: str = Field(min_length=2, max_length=256)
    reviewer_type: str = Field(pattern="^HUMAN$")
    sample_ids: list[str] = Field(min_length=1, max_length=500)
    notes: str = Field(default="", max_length=4000)



class AdminBatchConfirmation(BaseModel):
    admin_actor_id: str = Field(pattern="^admin$")
    reviewer_type: str = Field(pattern="^HUMAN$")
    sample_ids: list[str] = Field(min_length=1, max_length=500)
    confirmation_text: str = Field(pattern="^CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN$")
    admin_token: str | None = Field(default=None, max_length=512)
    vehicle_context_id: str | None = Field(default=None, max_length=256)
    video_id: str | None = Field(default=None, max_length=256)
    vehicle_id: str | None = Field(default=None, max_length=256)
    person_id: str | None = Field(default=None, max_length=256)
    camera_id: str | None = Field(default=None, max_length=256)
    conditions: list[str] = Field(default_factory=list, max_length=16)
    occupant_role: str | None = Field(
        default=None,
        pattern="^(driver|front_passenger|rear_left|rear_center|rear_right|other_occupant)$",
    )


def load_queue() -> list[dict]:
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else []


def load_latest_decisions() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                latest[item["sample_id"]] = item
    return latest


def load_acknowledged_ids() -> set[str]:
    acknowledged: set[str] = set()
    if acknowledgements_path.exists():
        for line in acknowledgements_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                acknowledged.update(json.loads(line).get("sample_ids", []))
    return acknowledged


def resolve_queue_image(item: dict) -> Path:
    path = Path(str(item.get("image_path", "")))
    if not path.is_file():
        parts = path.parts
        dataset_index = next(
            (index for index, value in enumerate(parts) if value.lower() == "datasets"),
            None,
        )
        if dataset_index is not None:
            relocated = datasets_root.joinpath(*parts[dataset_index + 1 :])
            if relocated.is_file():
                path = relocated
    if not path.is_file():
        sample_id = str(item["sample_id"]).split(":box:", 1)[0]
        matches = [
            candidate
            for candidate in (datasets_root / "incoming").rglob(f"{sample_id}.*")
            if candidate.is_file()
        ]
        if len(matches) != 1:
            raise HTTPException(404, "Image path is stale and sample_id fallback is ambiguous")
        path = matches[0]
    path = path.resolve()
    reviewed_root = datasets_root.resolve()
    if path != reviewed_root and reviewed_root not in path.parents:
        raise HTTPException(400, "Unsafe image path")
    expected = str(item.get("sha256") or item.get("source_sha256") or "").lower()
    if expected and sha256_file(path).lower() != expected:
        raise HTTPException(409, "SOURCE_HASH_MISMATCH")
    return path


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/queue")
def queue():
    samples = load_queue()
    latest = load_latest_decisions()
    if pending_only:
        samples = [item for item in samples if item["sample_id"] not in latest]
    return [
        {
            **item,
            "latest_review_decision": latest.get(item["sample_id"]),
        }
        for item in samples
    ]


@app.get("/api/status")
def status():
    samples = {item["sample_id"] for item in load_queue()}
    latest = {
        sample_id: item.get("status")
        for sample_id, item in load_latest_decisions().items()
        if sample_id in samples
    }
    acknowledged = load_acknowledged_ids() & samples
    status_values = (
        "APPROVED",
        "APPROVED_NEGATIVE",
        "REJECTED",
        "UNCERTAIN",
        *sorted(REVIEW1_STATUSES),
    )
    return {
        "total": len(samples),
        "decided": len(latest),
        "remaining": len(samples) - len(latest),
        "proposal_samples_acknowledged": len(acknowledged),
        "review1_decided": sum(value in REVIEW1_STATUSES for value in latest.values()),
        "human_confirmed": sum(
            value in {"APPROVED", "APPROVED_NEGATIVE"} for value in latest.values()
        ),
        "latest_status_counts": {
            value: sum(status == value for status in latest.values()) for value in status_values
        },
    }


@app.get("/api/image/{sample_id}")
def image(sample_id: str):
    item = next((entry for entry in load_queue() if entry["sample_id"] == sample_id), None)
    if not item:
        raise HTTPException(404, "Sample not found")
    return FileResponse(resolve_queue_image(item))


@app.post("/api/decision")
def decision(payload: Decision):
    queue_item = next(
        (item for item in load_queue() if item["sample_id"] == payload.sample_id), None
    )
    if queue_item is None:
        raise HTTPException(404, "Sample not found in the active queue")
    image_path = resolve_queue_image(queue_item)
    with decision_lock:
        previous = load_latest_decisions().get(payload.sample_id)
        record = payload.model_dump()
        if previous and all(previous.get(key) == value for key, value in record.items()):
            return {
                "saved": False,
                "duplicate": True,
                "decision_id": previous["decision_id"],
            }
        previous_status = (
            previous.get("new_status") or previous.get("status") if previous else "PENDING"
        )
        if previous_status not in {
            "PENDING",
            "APPROVED",
            "APPROVED_NEGATIVE",
            "REJECTED",
            "UNCERTAIN",
            *REVIEW1_STATUSES,
        }:
            raise HTTPException(409, f"Invalid previous review status: {previous_status}")
        for index, annotation in enumerate(record["annotations"]):
            annotation["box_id"] = annotation.get("box_id") or (f"{payload.sample_id}:box:{index}")
        if payload.reviewer_type == "AI" and payload.decision_reason == "PHONE_MISSING_LABEL":
            original_phone = sum(
                annotation.get("class_id") == 0
                for annotation in queue_item.get("annotations", [])
            )
            reviewed_phone = sum(
                annotation.get("class_id") == 0 for annotation in record["annotations"]
            )
            if reviewed_phone <= original_phone:
                raise HTTPException(400, "PHONE_MISSING_LABEL requires an added phone box")
        record["schema_version"] = 2
        record["previous_status"] = previous_status
        record["new_status"] = payload.status
        record["reviewed_at_utc"] = datetime.now(UTC).isoformat()
        record["proposal_review_snapshot"] = queue_item.get("proposal_review")
        record["source_sha256"] = sha256_file(image_path)
        record["source_group_id"] = queue_item.get("source_group_id")
        record["original_annotations"] = queue_item.get("annotations", [])
        record["reviewed_annotations"] = record["annotations"]
        record["source"] = {
            "queue_path": str(manifest_path),
            "queue_sha256": sha256_file(manifest_path),
            "source_id": queue_item.get("source_id"),
            "source_asset_id": queue_item.get("source_asset_id"),
        }
        record["decision_id"] = stable_json_hash(record)
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"saved": True, "decision_id": record["decision_id"]}


@app.post("/api/batch-acknowledgement")
def batch_acknowledgement(payload: BatchAcknowledgement):
    queue_by_id = {item["sample_id"]: item for item in load_queue()}
    unknown = sorted(set(payload.sample_ids) - queue_by_id.keys())
    if unknown:
        raise HTTPException(404, f"Samples not found in active queue: {unknown[:5]}")
    ineligible = []
    for sample_id in payload.sample_ids:
        proposal_review = queue_by_id[sample_id].get("proposal_review", {})
        if not (
            proposal_review.get("status") == "MODEL_ASSISTED_PROPOSAL_PENDING_APPROVAL"
            and proposal_review.get("decision") == "PASS"
            and proposal_review.get("tier") == "HIGH_CONFIDENCE"
        ):
            ineligible.append(sample_id)
    if ineligible:
        raise HTTPException(
            400,
            "Batch acknowledgement only accepts high-confidence model-assisted PASS samples; "
            f"ineligible: {ineligible[:5]}",
        )
    record = {
        "schema_version": 1,
        "reviewer_id": payload.reviewer_id,
        "reviewer_type": payload.reviewer_type,
        "status": "ADMIN_ACKNOWLEDGED_MODEL_PROPOSAL_BATCH",
        "admin_actor_id": payload.reviewer_id,
        "proposal_reviewer_id": REVIEWER_ID,
        "sample_ids": sorted(set(payload.sample_ids)),
        "notes": payload.notes,
        "acknowledged_at_utc": datetime.now(UTC).isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "queue_path": str(manifest_path),
        "queue_sha256": sha256_file(manifest_path),
        "batch_sha256": stable_json_hash(sorted(set(payload.sample_ids))),
        "review_policy_version": "review1_visual_policy_v1",
        "human_confirmation": False,
        "governance_eligible": False,
        "policy": (
            "Batch acknowledgement records model-assisted pending-approval proposals; "
            "it is not per-sample HUMAN_APPROVED."
        ),
    }
    with decision_lock:
        acknowledgements_path.parent.mkdir(parents=True, exist_ok=True)
        with acknowledgements_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"saved": True, "acknowledged": len(record["sample_ids"])}


def _confirmed_record(
    sample_id: str, previous: dict, queue_item: dict, payload: AdminBatchConfirmation
) -> dict:
    annotations = previous.get("reviewed_annotations", previous.get("annotations", []))
    target_status = "APPROVED" if annotations else "APPROVED_NEGATIVE"

    vehicle_context_id = (
        previous.get("vehicle_context_id")
        if previous.get("vehicle_context_id") and previous.get("vehicle_context_id") != "UNKNOWN"
        else payload.vehicle_context_id
    )
    video_id = (
        previous.get("video_id")
        if previous.get("video_id") and previous.get("video_id") != "UNKNOWN"
        else payload.video_id
    )
    vehicle_id = (
        previous.get("vehicle_id")
        if previous.get("vehicle_id") and previous.get("vehicle_id") != "UNKNOWN"
        else payload.vehicle_id
    )
    person_id = (
        previous.get("person_id")
        if previous.get("person_id") and previous.get("person_id") != "UNKNOWN"
        else payload.person_id
    )
    camera_id = (
        previous.get("camera_id")
        if previous.get("camera_id") and previous.get("camera_id") != "UNKNOWN"
        else payload.camera_id
    )
    conditions = previous.get("conditions") or payload.conditions
    occupant_role = (
        previous.get("occupant_role")
        if previous.get("occupant_role")
        not in {None, "PENDING", "UNCERTAIN", "unknown", "UNKNOWN"}
        else payload.occupant_role
    )

    required = {
        "vehicle_context_id": vehicle_context_id,
        "video_id": video_id,
        "vehicle_id": vehicle_id,
        "person_id": person_id,
        "camera_id": camera_id,
    }
    missing = [key for key, value in required.items() if not value or value == "UNKNOWN"]
    if missing or not conditions:
        raise HTTPException(
            409,
            f"Admin confirmation blocked for {sample_id}; incomplete governed metadata: {missing}",
        )
    if occupant_role in {None, "PENDING", "UNCERTAIN", "unknown", "UNKNOWN"}:
        raise HTTPException(409, f"Admin confirmation blocked for {sample_id}; unresolved role")
    unresolved = [
        index
        for index, annotation in enumerate(annotations)
        if annotation.get("occupant_role")
        in {None, "PENDING", "UNCERTAIN", "unknown", "UNKNOWN"}
    ]
    if unresolved:
        raise HTTPException(
            409,
            f"Admin confirmation blocked for {sample_id}; unresolved annotation roles: "
            f"{unresolved}",
        )
    record = {
        **previous,
        "schema_version": 3,
        "reviewer_id": "admin",
        "reviewer_type": "HUMAN",
        "previous_status": previous["new_status"],
        "new_status": target_status,
        "status": target_status,
        "decision_reason": "ADMIN_CONFIRMED_REVIEW1_PROPOSAL",
        "confirmation_source": "ADMIN_BATCH_CONFIRM",
        "confirmed_at_utc": datetime.now(UTC).isoformat(),
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "human_confirmation": True,
        "governance_eligible": True,
        "vehicle_context_id": vehicle_context_id,
        "video_id": video_id,
        "vehicle_id": vehicle_id,
        "person_id": person_id,
        "camera_id": camera_id,
        "conditions": conditions,
        "occupant_role": occupant_role,
        "source_group_id": queue_item.get("source_group_id"),
    }
    record.pop("decision_id", None)
    record["decision_id"] = stable_json_hash(record)
    return record


@app.post("/api/admin-batch-confirm")
def admin_batch_confirm(
    payload: AdminBatchConfirmation,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    expected_secret = os.environ.get("ADMIN_CONFIRMATION_SECRET") or os.environ.get("ADMIN_TOKEN")
    provided_token = payload.admin_token or x_admin_token
    if expected_secret:
        if not provided_token or provided_token != expected_secret:
            raise HTTPException(
                401, "UNAUTHORIZED_ADMIN_CONFIRMATION: Invalid or missing admin token"
            )

    queue_by_id = {item["sample_id"]: item for item in load_queue()}
    with decision_lock:
        latest = load_latest_decisions()
        records = []
        for sample_id in sorted(set(payload.sample_ids)):
            if sample_id not in queue_by_id:
                raise HTTPException(404, f"Sample not found in active queue: {sample_id}")
            previous = latest.get(sample_id)
            if not previous or previous.get("new_status") not in {
                "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION",
                "REVIEW1_ACCEPTED_PROPOSAL",
            }:
                raise HTTPException(
                    409, f"Sample is not an eligible Review 1 proposal: {sample_id}"
                )
            if previous.get("reviewer_type") != "AI" or previous.get("reviewer_id") != REVIEWER_ID:
                raise HTTPException(409, f"Invalid Review 1 provenance: {sample_id}")
            resolve_queue_image(queue_by_id[sample_id])
            records.append(_confirmed_record(sample_id, previous, queue_by_id[sample_id], payload))
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"saved": True, "confirmed": len(records), "reviewer_id": "admin"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=manifest_path)
    parser.add_argument("--decisions", type=Path, default=decisions_path)
    parser.add_argument("--acknowledgements", type=Path, default=acknowledgements_path)
    parser.add_argument(
        "--pending-only", action="store_true", help="Hide samples that already have any decision"
    )
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    manifest_path = args.manifest
    decisions_path = args.decisions
    acknowledgements_path = args.acknowledgements
    pending_only = args.pending_only
    uvicorn.run(app, host="127.0.0.1", port=args.port)
