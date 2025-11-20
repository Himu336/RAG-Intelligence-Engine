# app/router.py

from fastapi import APIRouter

# Feature routers
from app.interview_text.router import router as interview_text_router

router = APIRouter()

# Interview Text (Mock Interview System)
router.include_router(
    interview_text_router,
    prefix="/interview_text",
    tags=["Interview Text"]
)

# In future:
# router.include_router(chat_router, prefix="/chat", tags=["Chat"])
# router.include_router(admin_router, prefix="/admin", tags=["Admin"])
