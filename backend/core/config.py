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
    model_config_path: Path = Path("models/model_config.yaml")
    upload_dir: Path = Path("uploads")
    evidence_dir: Path = Path("evidence")
    max_upload_mb: int = 1024
    allow_camera_streams: bool = False
    cors_origins: str = "http://localhost:5173"
    access_token_minutes: int = 480

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
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

