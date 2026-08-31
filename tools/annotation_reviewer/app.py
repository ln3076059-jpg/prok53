from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Roadwatch Annotation Reviewer")
manifest_path = Path("datasets/manifests/review_queue.json")
decisions_path = Path("datasets/manifests/review_decisions.jsonl")
pending_only = False


class Annotation(BaseModel):
    class_id: int = Field(ge=0, le=2)
    yolo: tuple[float, float, float, float]
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(PENDING|driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|UNCERTAIN)$",
    )

    def model_post_init(self, __context) -> None:
        from training.common import YoloLabel

        YoloLabel(self.class_id, *self.yolo).validate()


class Decision(BaseModel):
    sample_id: str
    status: str = Field(pattern="^(APPROVED|REJECTED|UNCERTAIN)$")
    notes: str = Field(default="", max_length=4000)
    annotations: list[Annotation]
    vehicle_context_id: str | None = Field(default=None, max_length=256)
    occupant_role: str = Field(
        default="PENDING",
        pattern="^(PENDING|driver|front_passenger|rear_left|rear_center|rear_right|other_occupant|UNCERTAIN)$",
    )

    def model_post_init(self, __context) -> None:
        if self.status == "APPROVED" and not (self.vehicle_context_id or "").strip():
            raise ValueError("APPROVED samples require a vehicle_context_id")
        if self.status == "APPROVED" and self.occupant_role in {"PENDING", "UNCERTAIN"}:
            raise ValueError("APPROVED samples require a resolved occupant_role")
        if self.status == "APPROVED" and any(
            annotation.occupant_role in {"PENDING", "UNCERTAIN"}
            for annotation in self.annotations
        ):
            raise ValueError("APPROVED annotations require a resolved occupant_role per box")


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
    return {
        "total": len(samples),
        "decided": len(latest),
        "remaining": len(samples) - len(latest),
        "latest_status_counts": {
            value: sum(status == value for status in latest.values())
            for value in ("APPROVED", "REJECTED", "UNCERTAIN")
        },
    }


@app.get("/api/image/{sample_id}")
def image(sample_id: str):
    item = next((entry for entry in load_queue() if entry["sample_id"] == sample_id), None)
    if not item:
        raise HTTPException(404, "Sample not found")
    path = Path(item["image_path"]).resolve()
    reviewed_root = Path("datasets").resolve()
    if reviewed_root not in path.parents:
        raise HTTPException(400, "Unsafe image path")
    return FileResponse(path)


@app.post("/api/decision")
def decision(payload: Decision):
    if payload.sample_id not in {item["sample_id"] for item in load_queue()}:
        raise HTTPException(404, "Sample not found in the active queue")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as handle:
        record = payload.model_dump()
        record["reviewed_at_utc"] = datetime.now(UTC).isoformat()
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"saved": True}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=manifest_path)
    parser.add_argument("--decisions", type=Path, default=decisions_path)
    parser.add_argument(
        "--pending-only", action="store_true", help="Hide samples that already have any decision"
    )
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    manifest_path = args.manifest
    decisions_path = args.decisions
    pending_only = args.pending_only
    uvicorn.run(app, host="127.0.0.1", port=args.port)
