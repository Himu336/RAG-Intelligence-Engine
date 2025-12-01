# app/voice_interview/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from app.text_interview.schemas import (
    ExperienceLevel,
    InterviewType,
    DifficultyLevel
)


class InterviewRole(str, Enum):
    """AI Interviewer role types."""
    HR_RECRUITER = "HR Recruiter"
    TECHNICAL_INTERVIEWER = "Technical Interviewer"
    BEHAVIORAL_INTERVIEWER = "Behavioral Interviewer"
    MIXED_INTERVIEWER = "Mixed Interviewer"


class InterviewDifficulty(str, Enum):
    """Interview difficulty levels."""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class VoiceInterviewConfigRequest(BaseModel):
    """Request schema for creating a new voice interview."""
    job_role: str = Field(..., description="Target job role (e.g., Senior Software Engineer)")
    experience_level: ExperienceLevel = Field(..., description="Experience level")
    company: Optional[str] = Field(None, description="Target company (optional)")
    job_description: Optional[str] = Field(None, description="Job description text (optional)")
    interview_type: InterviewType = Field(..., description="Type of interview")
    interview_role: InterviewRole = Field(default=InterviewRole.MIXED_INTERVIEWER, description="AI interviewer role")
    difficulty: InterviewDifficulty = Field(default=InterviewDifficulty.INTERMEDIATE, description="Interview difficulty")
    user_id: str = Field(..., description="Unique user identifier")
    num_questions: int = Field(default=5, ge=3, le=15, description="Number of questions to generate")
    duration_minutes: int = Field(default=30, ge=10, le=60, description="Expected interview duration in minutes")


class TranscribeAudioRequest(BaseModel):
    """Request for transcribing audio."""
    session_id: str = Field(..., description="Interview session ID")
    audio_data: str = Field(..., description="Base64 encoded audio chunk")
    audio_format: str = Field(default="webm", description="Audio format (webm, wav, etc.)")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    is_final: bool = Field(default=True, description="Whether this is the final audio chunk for processing")


class ProcessMessageRequest(BaseModel):
    """Request for processing user message and generating AI response."""
    session_id: str = Field(..., description="Interview session ID")
    user_message: str = Field(..., description="User's transcribed message or text input")
    include_audio: bool = Field(default=True, description="Whether to include audio in response")
    stop_reason: Optional[str] = Field(default=None, description="Reason for stopping: 'manual', 'silence', 'timeout'")


class StartInterviewRequest(BaseModel):
    """Request to start an interview session."""
    session_id: str = Field(..., description="Interview session ID")


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class VoiceInterviewSessionResponse(BaseModel):
    """Response when creating a new voice interview session."""
    session_id: str = Field(..., description="Unique session identifier")
    total_questions: int = Field(..., description="Total number of questions")
    config: VoiceInterviewConfigRequest = Field(..., description="Interview configuration")
    created_at: datetime = Field(..., description="Session creation timestamp")


class TranscriptionResponse(BaseModel):
    """Response for transcribed user speech."""
    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    is_final: bool = Field(default=True, description="Whether transcription is final")


class AIResponse(BaseModel):
    """Response containing AI-generated text and optional audio."""
    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., description="AI response text")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio (if include_audio=true)")
    audio_format: str = Field(default="mp3", description="Audio format")
    current_phase: str = Field(..., description="Current interview phase")
    current_question_index: int = Field(..., description="Current question index")
    total_questions: int = Field(..., description="Total questions")
    progress_percentage: float = Field(..., description="Progress (0-100)")
    is_complete: bool = Field(..., description="Whether interview is complete")
    should_ask_followup: bool = Field(default=False, description="Whether a follow-up should be asked")


class InterviewState(BaseModel):
    """Current state of the interview."""
    session_id: str = Field(..., description="Session identifier")
    current_phase: str = Field(..., description="greeting, questions, followup, wrapup, complete")
    current_question_index: int = Field(..., description="Current question number (0-based)")
    total_questions: int = Field(..., description="Total questions")
    progress_percentage: float = Field(..., description="Progress (0-100)")
    is_complete: bool = Field(..., description="Whether interview is complete")
    is_started: bool = Field(..., description="Whether interview has started")


class VoiceInterviewAnalysis(BaseModel):
    """Complete analysis of the voice interview session."""
    session_id: str = Field(..., description="Session identifier")
    overall_score: float = Field(..., ge=0, le=100, description="Overall interview score out of 100")
    total_questions: int = Field(..., description="Total questions asked")
    answered_questions: int = Field(..., description="Number of questions answered")
    conversation_summary: str = Field(..., description="Summary of the conversation")
    strengths_summary: List[str] = Field(..., description="Overall strengths")
    improvement_areas: List[str] = Field(..., description="Areas to improve")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    communication_score: float = Field(..., ge=0, le=100, description="Communication/clarity score")
    content_score: float = Field(..., ge=0, le=100, description="Content/substance score")
    engagement_score: float = Field(..., ge=0, le=100, description="Engagement/presence score")
    completed_at: datetime = Field(..., description="Analysis generation timestamp")

