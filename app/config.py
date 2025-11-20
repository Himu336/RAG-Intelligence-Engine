# app/config.py

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Centralized application configuration loaded from environment variables."""

    # --- Gemini / LLM ---
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API Key")

    # --- SQL DB (Supabase / Neon / Postgres etc.) ---
    DATABASE_URL: str = Field(..., description="SQLAlchemy-compatible database URL")

    # --- Qdrant Vector Database ---
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant endpoint URL"
    )
    QDRANT_API_KEY: str | None = Field(
        default=None,
        description="Qdrant API key (for cloud deployments)"
    )

    PREDEFINED_COLLECTION: str = Field(
        default="predefined_context",
        description="Collection storing static predefined context"
    )
    USER_HISTORY_COLLECTION: str = Field(
        default="user_history",
        description="Collection storing long-term user memory"
    )

    # --- Redis Short-Term Memory ---
    REDIS_HOST: str = Field(
        ...,
        description="Redis Cloud hostname"
    )
    REDIS_PORT: int = Field(
        ...,
        description="Redis Cloud port"
    )
    REDIS_PASSWORD: str = Field(
        ...,
        description="Redis Cloud auth password"
    )

    # --- Chat TTL for short-term memory ---
    CHAT_TTL_SECONDS: int = Field(
        default=3600,
        description="Short-term chat memory expiry time in seconds"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance available everywhere
settings = Settings()
