from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from training.common import YoloLabel, sha256_file, stable_json_hash

app = FastAPI(title="Roadwatch Annotation Reviewer")
manifest_path = Path("datasets/manifests/review_queue.json")
decisions_path = Path("datasets/manifests/review_decisions.jsonl")
acknowledgements_path = Path("datasets/manifests/model_proposal_batch_acknowledgements.jsonl")
pending_only = False


class Annotation(BaseModel):
    box_id: str | None = Field(default=None, max_length=256)
    class_id: int = Field(ge=0, le=2)
    yolo: tuple[float, float, float, float]
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(PENDING|driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|UNCERTAIN)$",
    )

    def model_post_init(self, __context) -> None:
        YoloLabel(self.class_id, *self.yolo).validate()


class Decision(BaseModel):
    sample_id: str
    reviewer_id: str = Field(min_length=2, max_length=256)
    reviewer_type: str = Field(default="HUMAN", pattern="^HUMAN$")
    status: str = Field(pattern="^(APPROVED|APPROVED_NEGATIVE|REJECTED|UNCERTAIN)$")
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
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(PENDING|driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|UNCERTAIN)$",
    )

    def model_post_init(self, __context) -> None:
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
    sample_ids: list[str] = Field(min_length=1, max_length=500)
    notes: str = Field(default="", max_length=4000)


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


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/queue")
def queue():
    samples = load_queue()
    if pending_only:
        decided = load_latest_decisions()
        samples = [item for item in samples if item["sample_id"] not in decided]
    return samples


@app.get("/api/status")
def status():
    samples = {item["sample_id"] for item in load_queue()}
    latest = {
        sample_id: item.get("status")
        for sample_id, item in load_latest_decisions().items()
        if sample_id in samples
    }
    acknowledged = load_acknowledged_ids() & samples
    return {
        "total": len(samples),
        "decided": len(latest),
        "remaining": len(samples) - len(latest),
        "proposal_samples_acknowledged": len(acknowledged),
        "latest_status_counts": {
            value: sum(status == value for status in latest.values())
            for value in ("APPROVED", "APPROVED_NEGATIVE", "REJECTED", "UNCERTAIN")
        },
    }


@app.get("/api/image/{sample_id}")
def image(sample_id: str):
    item = next((entry for entry in load_queue() if entry["sample_id"] == sample_id), None)
    if not item:
        raise HTTPException(404, "Sample not found")
    path = Path(item["image_path"])
    if not path.is_file():
        matches = [
            candidate
            for candidate in Path("datasets/incoming").rglob(f"{sample_id}.*")
            if candidate.is_file()
        ]
        if len(matches) != 1:
            raise HTTPException(404, "Image path is stale and sample_id fallback is ambiguous")
        path = matches[0]
    path = path.resolve()
    reviewed_root = Path("datasets").resolve()
    if reviewed_root not in path.parents:
        raise HTTPException(400, "Unsafe image path")
    return FileResponse(path)


@app.post("/api/decision")
def decision(payload: Decision):
    queue_item = next(
        (item for item in load_queue() if item["sample_id"] == payload.sample_id), None
    )
    if queue_item is None:
        raise HTTPException(404, "Sample not found in the active queue")
    previous = load_latest_decisions().get(payload.sample_id)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as handle:
        record = payload.model_dump()
        for index, annotation in enumerate(record["annotations"]):
            annotation["box_id"] = annotation.get("box_id") or (f"{payload.sample_id}:box:{index}")
        record["schema_version"] = 2
        record["previous_status"] = (
            previous.get("new_status") or previous.get("status") if previous else "PENDING"
        )
        record["new_status"] = payload.status
        record["reviewed_at_utc"] = datetime.now(UTC).isoformat()
        record["proposal_review_snapshot"] = queue_item.get("proposal_review")
        record["source_sha256"] = queue_item.get("sha256")
        record["source_group_id"] = queue_item.get("source_group_id")
        record["source"] = {
            "queue_path": str(manifest_path),
            "queue_sha256": sha256_file(manifest_path),
            "source_id": queue_item.get("source_id"),
            "source_asset_id": queue_item.get("source_asset_id"),
        }
        record["decision_id"] = stable_json_hash(record)
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
        "reviewer_type": "HUMAN",
        "status": "ADMIN_ACKNOWLEDGED_MODEL_PROPOSAL_BATCH",
        "sample_ids": sorted(set(payload.sample_ids)),
        "notes": payload.notes,
        "acknowledged_at_utc": datetime.now(UTC).isoformat(),
        "queue_path": str(manifest_path),
        "queue_sha256": sha256_file(manifest_path),
        "governance_eligible": False,
        "policy": (
            "Batch acknowledgement records model-assisted pending-approval proposals; "
            "it is not per-sample HUMAN_APPROVED."
        ),
    }
    acknowledgements_path.parent.mkdir(parents=True, exist_ok=True)
    with acknowledgements_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"saved": True, "acknowledged": len(record["sample_ids"])}


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
