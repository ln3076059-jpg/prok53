from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = "development-only-change-before-production-1234"
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

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

