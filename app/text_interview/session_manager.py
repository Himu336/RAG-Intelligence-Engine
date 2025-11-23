# app/text_interview/session_manager.py

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from app.text_interview.schemas import (
    InterviewConfigRequest,
    InterviewQuestion,
    InterviewSessionResponse
)
import redis
from app.config import settings


class InterviewSessionManager:
    """
    Manages interview sessions using Redis for state persistence.
    Handles session creation, question tracking, and answer storage.
    """

    def __init__(self):
        """Initialize Redis connection for session management."""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Interview Session Manager: Redis connected")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None
            self._in_memory_sessions: Dict[str, dict] = {}
            self._session_ttl = timedelta(hours=24)  # 24 hour TTL

    def create_session(
        self,
        config: InterviewConfigRequest,
        questions: List[InterviewQuestion]
    ) -> InterviewSessionResponse:
        """
        Create a new interview session.
        
        Args:
            config: Interview configuration
            questions: Generated questions for the interview
            
        Returns:
            InterviewSessionResponse with session_id
        """
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        session_data = {
            "session_id": session_id,
            "config": config.model_dump(),
            "questions": [q.model_dump() for q in questions],
            "answers": {},  # {question_id: answer_text}
            "current_question_index": 0,
            "created_at": created_at.isoformat(),
            "completed_at": None,
            "is_complete": False
        }
        
        if self.redis_client:
            # Store in Redis with 24 hour TTL
            key = f"interview:session:{session_id}"
            import json
            self.redis_client.setex(
                key,
                86400,  # 24 hours in seconds
                json.dumps(session_data, default=str)
            )
        else:
            # In-memory fallback
            self._in_memory_sessions[session_id] = {
                **session_data,
                "_expires_at": created_at + self._session_ttl
            }
            # Clean expired sessions
            self._clean_expired_sessions()
        
        return InterviewSessionResponse(
            session_id=session_id,
            total_questions=len(questions),
            config=config,
            created_at=created_at
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data by session_id."""
        if self.redis_client:
            key = f"interview:session:{session_id}"
            import json
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        else:
            session = self._in_memory_sessions.get(session_id)
            if session:
                # Check expiration
                if datetime.utcnow() > session.get("_expires_at", datetime.max):
                    del self._in_memory_sessions[session_id]
                    return None
                # Remove internal fields
                return {k: v for k, v in session.items() if not k.startswith("_")}
            return None

    def get_current_question(self, session_id: str) -> Optional[InterviewQuestion]:
        """Get the current question for a session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        questions = session.get("questions", [])
        current_idx = session.get("current_question_index", 0)
        
        if current_idx >= len(questions):
            return None
        
        return InterviewQuestion(**questions[current_idx])

    def submit_answer(
        self,
        session_id: str,
        question_id: int,
        answer: str
    ) -> bool:
        """
        Submit an answer for a specific question.
        
        Args:
            session_id: Session identifier
            question_id: Question index (0-based)
            answer: User's answer text
            
        Returns:
            True if answer was saved successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Validate question_id
        questions = session.get("questions", [])
        if question_id < 0 or question_id >= len(questions):
            return False
        
        # Store answer
        session["answers"][str(question_id)] = answer.strip()
        
        # Update current question index if this is the current question
        current_idx = session.get("current_question_index", 0)
        if question_id == current_idx:
            session["current_question_index"] = current_idx + 1
        
        # Check if complete
        total_questions = len(questions)
        if session["current_question_index"] >= total_questions:
            session["is_complete"] = True
            session["completed_at"] = datetime.utcnow().isoformat()
        
        # Save back to storage
        if self.redis_client:
            key = f"interview:session:{session_id}"
            import json
            self.redis_client.setex(
                key,
                86400,
                json.dumps(session, default=str)
            )
        else:
            self._in_memory_sessions[session_id] = session
        
        return True

    def get_all_answers(self, session_id: str) -> Dict[int, str]:
        """Get all submitted answers for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        answers = session.get("answers", {})
        # Convert string keys to int
        return {int(k): v for k, v in answers.items()}

    def is_complete(self, session_id: str) -> bool:
        """Check if interview session is complete."""
        session = self.get_session(session_id)
        if not session:
            return False
        return session.get("is_complete", False)

    def get_progress(self, session_id: str) -> tuple[int, int, float]:
        """
        Get interview progress.
        
        Returns:
            (current_question_number, total_questions, progress_percentage)
        """
        session = self.get_session(session_id)
        if not session:
            return (0, 0, 0.0)
        
        total = len(session.get("questions", []))
        current_idx = session.get("current_question_index", 0)
        current_num = current_idx + 1 if current_idx < total else total
        
        progress = (current_num / total * 100) if total > 0 else 0.0
        
        return (current_num, total, progress)

    def _clean_expired_sessions(self):
        """Clean up expired in-memory sessions."""
        if not self._in_memory_sessions:
            return
        
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self._in_memory_sessions.items()
            if now > session.get("_expires_at", datetime.max)
        ]
        
        for sid in expired:
            del self._in_memory_sessions[sid]

