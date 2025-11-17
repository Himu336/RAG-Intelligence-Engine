# app/schemas.py
from pydantic import BaseModel

class RAGRequest(BaseModel):
    message: str
    user_id: str

class RAGResponse(BaseModel):
    ai_text: str
