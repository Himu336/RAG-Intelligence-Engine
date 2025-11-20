# app/db/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings


# ---------------------------------------------------
# Validate environment configuration
# ---------------------------------------------------
DATABASE_URL = settings.DATABASE_URL
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL is missing. Set it in your environment or .env file.")


# ---------------------------------------------------
# SQLAlchemy Engine & Session
# ---------------------------------------------------
# Using synchronous SQLAlchemy now (FastAPI + Python service is sync)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # validates stale connections
    pool_size=5,
    max_overflow=10
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ---------------------------------------------------
# Base class for all Models
# ---------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------
# FastAPI Dependency (used inside endpoints)
# ---------------------------------------------------
def get_db():
    """
    FastAPI dependency:
        from app.db.connection import get_db
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
