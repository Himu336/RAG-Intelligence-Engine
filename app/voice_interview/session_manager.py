# app/voice_interview/session_manager.py

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from app.voice_interview.schemas import (
    VoiceInterviewConfigRequest,
    VoiceInterviewSessionResponse,
    InterviewState
)
from app.text_interview.schemas import InterviewQuestion
import redis
from app.config import settings


class VoiceInterviewSessionManager:
    """
    Manages voice interview sessions using Redis for state persistence.
    Handles session creation, conversation tracking, and state management.
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
            self.redis_client.ping()
            print("✅ Voice Interview Session Manager: Redis connected")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None
            self._in_memory_sessions: Dict[str, dict] = {}
            self._session_ttl = timedelta(hours=24)

    def create_session(
        self,
        config: VoiceInterviewConfigRequest,
        questions: List[InterviewQuestion]
    ) -> VoiceInterviewSessionResponse:
        """
        Create a new voice interview session.
        
        Args:
            config: Interview configuration
            questions: Generated questions for the interview
            
        Returns:
            VoiceInterviewSessionResponse with session_id
        """
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        session_data = {
            "session_id": session_id,
            "config": config.model_dump(),
            "questions": [q.model_dump() for q in questions],
            "conversation_turns": [],
            "current_phase": "greeting",
            "current_question_index": 0,
            "followup_count": 0,
            "max_followups_per_question": 2,
            "created_at": created_at.isoformat(),
            "started_at": None,
            "completed_at": None,
            "is_complete": False,
            "is_started": False,
            "pending_final_transcription": False,
            "last_final_transcript": None
        }
        
        if self.redis_client:
            key = f"voice_interview:session:{session_id}"
            self.redis_client.setex(
                key,
                86400,
                json.dumps(session_data, default=str)
            )
        else:
            self._in_memory_sessions[session_id] = {
                **session_data,
                "_expires_at": created_at + self._session_ttl
            }
            self._clean_expired_sessions()
        
        return VoiceInterviewSessionResponse(
            session_id=session_id,
            total_questions=len(questions),
            config=config,
            created_at=created_at
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data by session_id."""
        if self.redis_client:
            key = f"voice_interview:session:{session_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        else:
            session = self._in_memory_sessions.get(session_id)
            if session:
                if datetime.utcnow() > session.get("_expires_at", datetime.max):
                    del self._in_memory_sessions[session_id]
                    return None
                return {k: v for k, v in session.items() if not k.startswith("_")}
            return None

    def start_session(self, session_id: str) -> bool:
        """Mark session as started."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["is_started"] = True
        session["started_at"] = datetime.utcnow().isoformat()
        session["current_phase"] = "greeting"
        
        self._save_session(session_id, session)
        return True

    def add_conversation_turn(
        self,
        session_id: str,
        speaker: str,
        text: str,
        stop_reason: Optional[str] = None
    ) -> bool:
        """
        Add a conversation turn to the session.
        
        Args:
            session_id: Session identifier
            speaker: "user" or "ai"
            text: Transcribed text or AI response
            stop_reason: Optional reason for stopping (manual, silence, timeout)
            
        Returns:
            True if added successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        turns = session.get("conversation_turns", [])
        turn = {
            "turn_id": len(turns),
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        }
        if stop_reason:
            turn["stop_reason"] = stop_reason
        
        turns.append(turn)
        session["conversation_turns"] = turns
        
        self._save_session(session_id, session)
        return True

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history as list of dicts."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        return session.get("conversation_turns", [])

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

    def advance_to_next_question(self, session_id: str) -> bool:
        """Advance to the next question."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        questions = session.get("questions", [])
        current_idx = session.get("current_question_index", 0)
        
        if current_idx >= len(questions) - 1:
            session["current_phase"] = "wrapup"
            session["current_question_index"] = len(questions)
        else:
            session["current_question_index"] = current_idx + 1
            session["current_phase"] = "questions"
            session["followup_count"] = 0
        
        self._save_session(session_id, session)
        return True

    def increment_followup(self, session_id: str) -> bool:
        """Increment follow-up counter for current question."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["followup_count"] = session.get("followup_count", 0) + 1
        session["current_phase"] = "followup"
        
        self._save_session(session_id, session)
        return True

    def can_ask_followup(self, session_id: str) -> bool:
        """Check if we can ask a follow-up for the current question."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        followup_count = session.get("followup_count", 0)
        max_followups = session.get("max_followups_per_question", 2)
        
        return followup_count < max_followups

    def set_phase(self, session_id: str, phase: str) -> bool:
        """Set the current interview phase."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        valid_phases = ["greeting", "questions", "followup", "wrapup", "complete"]
        if phase not in valid_phases:
            return False
        
        session["current_phase"] = phase
        
        if phase == "complete":
            session["is_complete"] = True
            session["completed_at"] = datetime.utcnow().isoformat()
        
        self._save_session(session_id, session)
        return True

    def get_state(self, session_id: str) -> Optional[InterviewState]:
        """Get current interview state."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        total = len(session.get("questions", []))
        current_idx = session.get("current_question_index", 0)
        
        # Zero-based progress: current_idx / total * 100
        progress = (current_idx / total * 100) if total > 0 else 0.0
        
        return InterviewState(
            session_id=session_id,
            current_phase=session.get("current_phase", "greeting"),
            current_question_index=current_idx,
            total_questions=total,
            progress_percentage=round(progress, 2),
            is_complete=session.get("is_complete", False),
            is_started=session.get("is_started", False)
        )

    def is_complete(self, session_id: str) -> bool:
        """Check if interview session is complete."""
        session = self.get_session(session_id)
        if not session:
            return False
        return session.get("is_complete", False)

    def _save_session(self, session_id: str, session_data: dict):
        """Save session data to storage."""
        if self.redis_client:
            key = f"voice_interview:session:{session_id}"
            self.redis_client.setex(
                key,
                86400,
                json.dumps(session_data, default=str)
            )
        else:
            self._in_memory_sessions[session_id] = session_data

    def set_final_transcription(self, session_id: str, transcript: str) -> bool:
        """
        Mark that a final transcription has been completed.
        
        Args:
            session_id: Session identifier
            transcript: The final transcribed text
            
        Returns:
            True if set successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["pending_final_transcription"] = False
        session["last_final_transcript"] = transcript
        
        self._save_session(session_id, session)
        return True
    
    def mark_pending_final_transcription(self, session_id: str) -> bool:
        """
        Mark that we're waiting for a final transcription.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if marked successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["pending_final_transcription"] = True
        
        self._save_session(session_id, session)
        return True
    
    def has_pending_final_transcription(self, session_id: str) -> bool:
        """
        Check if there's a pending final transcription.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if waiting for final transcription
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        return session.get("pending_final_transcription", False)
    
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

