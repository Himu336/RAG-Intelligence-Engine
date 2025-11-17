from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str

    # Qdrant settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    PREDEFINED_COLLECTION: str = "predefined_context"
    USER_HISTORY_COLLECTION: str = "user_history"

    # Redis Cloud settings (NEW)
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str

    # Short-term chat memory TTL (default 1 hour)
    CHAT_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
