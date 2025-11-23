# app/text_interview/router.py

from fastapi import APIRouter, HTTPException, status
from typing import Optional
from app.text_interview.schemas import (
    InterviewConfigRequest,
    InterviewSessionResponse,
    QuestionResponse,
    AnswerSubmissionRequest,
    AnswerSubmissionResponse,
    InterviewAnalysis
)
from app.text_interview.generator import InterviewQuestionGenerator
from app.text_interview.session_manager import InterviewSessionManager
from app.text_interview.analyzer import InterviewAnalyzer

# Initialize router
router = APIRouter(
    prefix="/interview",
    tags=["Interview"]
)

# Initialize services
question_generator = InterviewQuestionGenerator()
session_manager = InterviewSessionManager()
analyzer = InterviewAnalyzer()


@router.post("/create", response_model=InterviewSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_interview(config: InterviewConfigRequest):
    """
    Create a new interview session with generated questions.
    
    This endpoint:
    1. Takes interview configuration (job role, experience level, etc.)
    2. Generates relevant questions using AI
    3. Creates a session and returns session_id
    
    Your Node.js server should call this when user clicks "Start Interview".
    """
    try:
        # Generate questions based on configuration
        questions = question_generator.generate_questions(config)
        
        if not questions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate interview questions"
            )
        
        # Create session
        session_response = session_manager.create_session(config, questions)
        
        return session_response
        
    except Exception as e:
        print(f"❌ Error creating interview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create interview session: {str(e)}"
        )


@router.get("/{session_id}/question", response_model=QuestionResponse)
async def get_current_question(session_id: str):
    """
    Get the current question for an interview session.
    
    Returns:
    - Current question with metadata (difficulty, hint, etc.)
    - Progress information (question number, total, percentage)
    - Whether interview is complete
    
    Your Node.js server should call this to display the current question.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Interview session '{session_id}' not found"
        )
    
    # Check if complete
    is_complete = session_manager.is_complete(session_id)
    
    if is_complete:
        raise HTTPException(
            status_code=400,
            detail="Interview is already complete. Use /analysis endpoint to get results."
        )
    
    # Get current question
    current_question = session_manager.get_current_question(session_id)
    
    if not current_question:
        raise HTTPException(
            status_code=400,
            detail="No more questions available"
        )
    
    # Get progress
    current_num, total, progress = session_manager.get_progress(session_id)
    
    return QuestionResponse(
        session_id=session_id,
        current_question=current_question,
        question_number=current_num,
        total_questions=total,
        progress_percentage=round(progress, 2),
        is_complete=False
    )


@router.post("/{session_id}/answer", response_model=AnswerSubmissionResponse)
async def submit_answer(session_id: str, request: AnswerSubmissionRequest):
    """
    Submit an answer for the current question.
    
    This endpoint:
    1. Saves the user's answer
    2. Advances to the next question
    3. Returns whether there are more questions
    
    Your Node.js server should call this when user clicks "Next Question".
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Interview session '{session_id}' not found"
        )
    
    if session_manager.is_complete(session_id):
        raise HTTPException(
            status_code=400,
            detail="Interview is already complete"
        )
    
    # Validate answer
    if not request.answer or not request.answer.strip():
        raise HTTPException(
            status_code=400,
            detail="Answer cannot be empty"
        )
    
    # Submit answer
    success = session_manager.submit_answer(
        session_id=session_id,
        question_id=request.question_id,
        answer=request.answer
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to submit answer. Invalid question_id or session state."
        )
    
    # Check if complete after submission
    is_complete = session_manager.is_complete(session_id)
    
    # Check if there's a next question
    next_question = session_manager.get_current_question(session_id)
    next_available = next_question is not None and not is_complete
    
    return AnswerSubmissionResponse(
        session_id=session_id,
        question_id=request.question_id,
        answer_saved=True,
        next_question_available=next_available,
        is_complete=is_complete
    )


@router.get("/{session_id}/analysis", response_model=InterviewAnalysis)
async def get_analysis(session_id: str):
    """
    Get comprehensive analysis of the completed interview.
    
    This endpoint:
    1. Evaluates all submitted answers
    2. Provides individual question feedback
    3. Generates overall analysis with strengths, improvements, and recommendations
    
    Your Node.js server should call this when interview is complete to show results.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Interview session '{session_id}' not found"
        )
    
    # Get questions and answers
    questions_data = session.get("questions", [])
    answers = session_manager.get_all_answers(session_id)
    
    if not questions_data:
        raise HTTPException(
            status_code=400,
            detail="No questions found in session"
        )
    
    # Convert questions data to InterviewQuestion objects
    from app.text_interview.schemas import InterviewQuestion, DifficultyLevel, QuestionHint
    questions = []
    for q_data in questions_data:
        hint_data = q_data.get("hint")
        hint = None
        if hint_data:
            hint = QuestionHint(**hint_data) if isinstance(hint_data, dict) else None
        
        questions.append(InterviewQuestion(
            question_id=q_data["question_id"],
            question_text=q_data["question_text"],
            difficulty=DifficultyLevel(q_data["difficulty"]),
            hint=hint,
            interview_type=q_data.get("interview_type", "Mixed")
        ))
    
    # Generate analysis
    try:
        analysis = analyzer.generate_analysis(
            session_id=session_id,
            questions=questions,
            answers=answers,
            config=session.get("config", {})
        )
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error generating analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analysis: {str(e)}"
        )


@router.get("/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current status of an interview session.
    
    Returns basic information about the session state.
    Useful for checking if session exists and its completion status.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Interview session '{session_id}' not found"
        )
    
    current_num, total, progress = session_manager.get_progress(session_id)
    is_complete = session_manager.is_complete(session_id)
    answers = session_manager.get_all_answers(session_id)
    
    return {
        "session_id": session_id,
        "is_complete": is_complete,
        "current_question": current_num,
        "total_questions": total,
        "progress_percentage": round(progress, 2),
        "answers_submitted": len(answers),
        "created_at": session.get("created_at")
    }

