from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    PREDEFINED_COLLECTION: str = "predefined_context"
    USER_HISTORY_COLLECTION: str = "user_history"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
