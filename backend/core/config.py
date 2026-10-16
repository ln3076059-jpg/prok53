from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEVELOPMENT_SECRET_KEY = "development-only-change-before-production-1234"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = DEVELOPMENT_SECRET_KEY
    database_url: str = "sqlite:///./driver_safety.db"
    model_path: Path = Path("models/active/best.pt")
    model_config_path: Path = Path("models/model_config_v2.yaml")
    upload_dir: Path = Path("uploads")
    evidence_dir: Path = Path("evidence")
    research_phone_review_source: Path = Path("reports/v9_research/phone_proposal_mining.json")
    research_phone_review_dir: Path = Path("reports/v9_research/human_phone_reviews")
    max_upload_mb: int = 1024
    allow_camera_streams: bool = False
    cors_origins: str = "http://localhost:5173"
    access_token_minutes: int = 480
    analysis_queue_backend: str = "background"

    @field_validator("app_env")
    @classmethod
    def supported_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"development", "testing", "production"}:
            raise ValueError("APP_ENV must be development, testing, or production")
        return normalized

    @field_validator("analysis_queue_backend")
    @classmethod
    def supported_queue_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"background", "database"}:
            raise ValueError("ANALYSIS_QUEUE_BACKEND must be background or database")
        return normalized

    @field_validator("secret_key")
    @classmethod
    def secure_production_key(cls, value: str, info):
        if len(value) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return value

    @model_validator(mode="after")
    def reject_development_secret_in_production(self):
        if (
            self.app_env.strip().lower() == "production"
            and self.secret_key == DEVELOPMENT_SECRET_KEY
        ):
            raise ValueError("production cannot use the public development SECRET_KEY")
        if self.app_env == "production" and self.secret_key.lower().startswith("replace-"):
            raise ValueError("production cannot use a placeholder SECRET_KEY")
        if self.app_env == "production" and "replace-me" in self.database_url.lower():
            raise ValueError("production cannot use placeholder database credentials")
        if self.app_env == "production" and self.analysis_queue_backend != "database":
            raise ValueError("production requires ANALYSIS_QUEUE_BACKEND=database")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
