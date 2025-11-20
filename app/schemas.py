# app/schemas.py

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Incoming request for the Personal Coach (RAG) endpoint."""
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="Unique user identifier")


class RAGResponse(BaseModel):
    """Response returned by the Personal Coach."""
    ai_text: str = Field(..., description="Generated AI response")
