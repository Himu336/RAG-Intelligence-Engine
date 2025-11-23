# app/text_interview/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ExperienceLevel(str, Enum):
    """Experience level options for interview configuration."""
    ENTRY = "Entry Level (0-2 years)"
    MID = "Mid Level (3-5 years)"
    SENIOR = "Senior Level (6-10 years)"
    EXECUTIVE = "Executive Level (10+ years)"


class InterviewType(str, Enum):
    """Interview type options."""
    BEHAVIORAL = "Behavioral"
    TECHNICAL = "Technical"
    SYSTEM_DESIGN = "System Design"
    MIXED = "Mixed"


class DifficultyLevel(str, Enum):
    """Question difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class InterviewConfigRequest(BaseModel):
    """Request schema for creating a new interview."""
    job_role: str = Field(..., description="Target job role (e.g., Senior Software Engineer)")
    experience_level: ExperienceLevel = Field(..., description="Experience level")
    company: Optional[str] = Field(None, description="Target company (optional)")
    job_description: Optional[str] = Field(None, description="Job description text (optional)")
    interview_type: InterviewType = Field(..., description="Type of interview")
    user_id: str = Field(..., description="Unique user identifier")
    num_questions: int = Field(default=5, ge=3, le=15, description="Number of questions to generate")


class AnswerSubmissionRequest(BaseModel):
    """Request schema for submitting an answer."""
    answer: str = Field(..., description="User's answer text")
    question_id: int = Field(..., description="Question index (0-based)")


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class QuestionHint(BaseModel):
    """Hint/guidance for a question."""
    text: str = Field(..., description="Hint text")
    framework: Optional[str] = Field(None, description="Suggested framework (e.g., STAR)")


class InterviewQuestion(BaseModel):
    """Single interview question with metadata."""
    question_id: int = Field(..., description="Question index (0-based)")
    question_text: str = Field(..., description="The interview question")
    difficulty: DifficultyLevel = Field(..., description="Question difficulty")
    hint: Optional[QuestionHint] = Field(None, description="Optional hint for the question")
    interview_type: str = Field(..., description="Type of question (behavioral, technical, etc.)")


class InterviewSessionResponse(BaseModel):
    """Response when creating a new interview session."""
    session_id: str = Field(..., description="Unique session identifier")
    total_questions: int = Field(..., description="Total number of questions")
    config: InterviewConfigRequest = Field(..., description="Interview configuration")
    created_at: datetime = Field(..., description="Session creation timestamp")


class QuestionResponse(BaseModel):
    """Response for getting the current/next question."""
    session_id: str = Field(..., description="Session identifier")
    current_question: InterviewQuestion = Field(..., description="Current question")
    question_number: int = Field(..., description="Current question number (1-based)")
    total_questions: int = Field(..., description="Total questions in interview")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    is_complete: bool = Field(..., description="Whether interview is complete")


class AnswerSubmissionResponse(BaseModel):
    """Response after submitting an answer."""
    session_id: str = Field(..., description="Session identifier")
    question_id: int = Field(..., description="Question that was answered")
    answer_saved: bool = Field(..., description="Whether answer was saved successfully")
    next_question_available: bool = Field(..., description="Whether there are more questions")
    is_complete: bool = Field(..., description="Whether interview is complete")


class AnswerEvaluation(BaseModel):
    """Evaluation of a single answer."""
    question_id: int = Field(..., description="Question index")
    question_text: str = Field(..., description="Question that was asked")
    user_answer: str = Field(..., description="User's submitted answer")
    strengths: List[str] = Field(..., description="Strengths in the answer")
    improvements: List[str] = Field(..., description="Areas for improvement")
    score: float = Field(..., ge=0, le=100, description="Score out of 100")
    feedback: str = Field(..., description="Detailed feedback with checkmarks and bullet points")


class InterviewAnalysis(BaseModel):
    """Complete analysis of the interview session."""
    session_id: str = Field(..., description="Session identifier")
    overall_score: float = Field(..., ge=0, le=100, description="Overall interview score out of 100")
    total_questions: int = Field(..., description="Total questions answered")
    answered_questions: int = Field(..., description="Number of questions with answers")
    evaluations: List[AnswerEvaluation] = Field(..., description="Individual question evaluations")
    overall_feedback: str = Field(..., description="Overall interview feedback")
    strengths_summary: List[str] = Field(..., description="Overall strengths")
    improvement_areas: List[str] = Field(..., description="Areas to improve")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    completed_at: datetime = Field(..., description="Analysis generation timestamp")

