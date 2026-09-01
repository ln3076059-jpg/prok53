from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.models.entities import EventType, InputScope, JobStatus, ReviewStatus


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class VideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    original_name: str
    size_bytes: int
    mime_type: str
    input_scope: InputScope
    created_at: datetime


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    video_id: str
    status: JobStatus
    progress: float
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    video_id: str
    event_type: EventType
    confidence: float
    frame_number: int
    timestamp_seconds: float
    track_id: int | None
    occupant_role: str
    vehicle_context_id: str
    model_version: str
    review_status: ReviewStatus
    created_at: datetime


class EvidenceLinks(BaseModel):
    available: bool = False
    original_url: str | None = None
    annotated_url: str | None = None
    clip_url: str | None = None
    trace_url: str | None = None
    hashes: dict[str, str] = Field(default_factory=dict)


class ReviewHistoryResponse(BaseModel):
    previous_status: str
    new_status: str
    notes: str | None
    reviewer_id: str
    created_at: datetime


class EventDetailResponse(EventResponse):
    vehicle_type: str = "unknown"
    fusion_score: float | None = None
    temporal_score: float | None = None
    evidence: EvidenceLinks = Field(default_factory=EvidenceLinks)
    evidence_trace: dict[str, Any] = Field(default_factory=dict)
    review_history: list[ReviewHistoryResponse] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    status: ReviewStatus
    notes: str | None = Field(default=None, max_length=2000)
