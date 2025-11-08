from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    industry: str = Field(default="Tollywood", alias="INDUSTRY")
    max_articles: int = Field(default=10, alias="MAX_ARTICLES", ge=1, le=50)
    news_sources: List[str] = Field(default_factory=lambda: ["google_news"], alias="NEWS_SOURCES")

    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    image_model: str = Field(default="stable-diffusion", alias="IMAGE_MODEL")
    image_output_size: int = Field(default=1080, alias="IMAGE_OUTPUT_SIZE")

    sqlite_db_path: Path = Field(
        default=Path("data") / "automation.db", alias="SQLITE_DB_PATH"
    )
    output_dir: Path = Field(default=Path("output"), alias="OUTPUT_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @validator("news_sources")
    def _normalize_sources(cls, value: List[str]) -> List[str]:
        return [source.strip().lower() for source in value]

    @validator("sqlite_db_path", "output_dir", pre=True)
    def _expand_path(cls, value: Optional[str | os.PathLike[str]]) -> Path:
        if value is None:
            return value
        return Path(value).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


class NewsArticle(BaseModel):
    """Structured representation of a news article."""

    title: str
    url: str
    source: Optional[str] = None
    summary: Optional[str] = None
    published_at: Optional[str] = None


class ViralIdea(BaseModel):
    """LLM-generated social post idea."""

    headline: str
    angle: str
    key_points: List[str]
    source_url: str


class InstagramAsset(BaseModel):
    """Represents image and caption outputs."""

    idea: ViralIdea
    caption: str
    image_path: Path


