from datetime import datetime

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


class ReviewRequest(BaseModel):
    status: ReviewStatus
    notes: str | None = Field(default=None, max_length=2000)
