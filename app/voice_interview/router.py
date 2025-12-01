# app/voice_interview/router.py

from fastapi import APIRouter, HTTPException, status
from app.voice_interview.schemas import (
    VoiceInterviewConfigRequest,
    VoiceInterviewSessionResponse,
    VoiceInterviewAnalysis,
    InterviewState,
    TranscriptionResponse,
    AIResponse,
    TranscribeAudioRequest,
    ProcessMessageRequest,
    StartInterviewRequest
)
from app.text_interview.generator import InterviewQuestionGenerator
from app.text_interview.schemas import InterviewConfigRequest
from app.voice_interview.session_manager import VoiceInterviewSessionManager
from app.voice_interview.analyzer import VoiceInterviewAnalyzer
from app.voice_interview.conversation_manager import VoiceConversationManager
from app.voice_interview.speech_to_text import SpeechToTextService, TextToSpeechService

# Initialize router
router = APIRouter(
    prefix="/voice-interview",
    tags=["Voice Interview"]
)

# Initialize services
question_generator = InterviewQuestionGenerator()
session_manager = VoiceInterviewSessionManager()
analyzer = VoiceInterviewAnalyzer()
conversation_manager = VoiceConversationManager()
speech_to_text = SpeechToTextService()
text_to_speech = TextToSpeechService()


@router.post("/create", response_model=VoiceInterviewSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_voice_interview(config: VoiceInterviewConfigRequest):
    """
    Create a new voice interview session with generated questions.
    
    Node.js gateway should call this when user wants to start an interview.
    Returns session_id that can be used for subsequent API calls.
    """
    try:
        # Convert voice interview config to text interview config for question generation
        text_config = InterviewConfigRequest(
            job_role=config.job_role,
            experience_level=config.experience_level,
            company=config.company,
            job_description=config.job_description,
            interview_type=config.interview_type,
            user_id=config.user_id,
            num_questions=config.num_questions
        )
        
        # Generate questions
        questions = question_generator.generate_questions(text_config)
        
        if not questions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate interview questions"
            )
        
        # Create session
        session_response = session_manager.create_session(config, questions)
        
        return session_response
        
    except Exception as e:
        print(f"❌ Error creating voice interview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create voice interview session: {str(e)}"
        )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscribeAudioRequest):
    """
    Transcribe audio chunk to text.
    
    Node.js gateway should call this when it receives audio from the client.
    - If is_final == false → returns early with partial transcript (for preview)
    - If is_final == true → performs full ElevenLabs STT and marks transcription as complete
    
    Returns transcribed text with confidence score.
    """
    try:
        # If this is not a final chunk, return early with partial transcript
        # This prevents incorrect transcription due to small chunks
        if not request.is_final:
            # Mark that we're waiting for final transcription
            session_manager.mark_pending_final_transcription(request.session_id)
            
            return TranscriptionResponse(
                session_id=request.session_id,
                text="",  # Empty for partial chunks - frontend handles preview
                confidence=0.0,
                is_final=False
            )
        
        # Only process final audio chunks
        transcript, confidence = speech_to_text.transcribe_base64(
            request.audio_data,
            sample_rate=request.sample_rate,
            audio_format=request.audio_format
        )
        
        # Mark final transcription as complete
        session_manager.set_final_transcription(request.session_id, transcript)
        
        return TranscriptionResponse(
            session_id=request.session_id,
            text=transcript,
            confidence=confidence,
            is_final=True
        )
    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )


@router.post("/start", response_model=AIResponse)
async def start_interview(request: StartInterviewRequest):
    """
    Start the interview - generates and returns greeting.
    
    Node.js gateway should call this when user clicks "Start Interview".
    Returns greeting text and optional audio.
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found"
        )
    
    # Mark as started
    session_manager.start_session(request.session_id)
    
    # Generate greeting
    config_data = session.get("config", {})
    from app.voice_interview.schemas import VoiceInterviewConfigRequest
    config = VoiceInterviewConfigRequest(**config_data)
    
    greeting = conversation_manager.generate_greeting(
        config, config.interview_role
    )
    
    # Add greeting to conversation
    session_manager.add_conversation_turn(request.session_id, "ai", greeting)
    
    # Generate audio (async)
    audio_base64 = await text_to_speech.synthesize_to_base64_async(
        text=greeting,
        language_code="en-US",
        voice_name=None,  # Uses default voice based on gender
        ssml_gender="NEUTRAL",
        audio_format="mp3"
    )
    
    # Get state
    state = session_manager.get_state(request.session_id)
    
    return AIResponse(
        session_id=request.session_id,
        text=greeting,
        audio_data=audio_base64,
        audio_format="mp3",
        current_phase=state.current_phase if state else "greeting",
        current_question_index=state.current_question_index if state else 0,
        total_questions=state.total_questions if state else 0,
        progress_percentage=state.progress_percentage if state else 0.0,
        is_complete=state.is_complete if state else False,
        should_ask_followup=False
    )


@router.post("/process", response_model=AIResponse)
async def process_message(request: ProcessMessageRequest):
    """
    Process user message and generate AI response.
    
    Node.js gateway should call this after transcribing user audio or receiving text.
    This is the main endpoint for conversation flow.
    Returns AI response text and optional audio.
    
    IMPORTANT: This endpoint will only process messages after final transcription is complete.
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found"
        )
    
    # Enforce ordering: reject if final transcription is still pending
    if session_manager.has_pending_final_transcription(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot process message: waiting for final transcription to complete. Please wait for final transcript before calling /process."
        )
    
    # Reject empty or partial user messages
    if not request.user_message or len(request.user_message.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="User message cannot be empty. Please provide a valid transcribed message."
        )
    
    # Reject messages with less than 2 words (likely partial transcription)
    word_count = len(request.user_message.split())
    if word_count < 2:
        raise HTTPException(
            status_code=400,
            detail=f"User message is too short ({word_count} word(s)). Please wait for complete transcription with at least 2 words."
        )
    
    # Add user turn to conversation (with stop_reason if provided)
    session_manager.add_conversation_turn(
        request.session_id, 
        "user", 
        request.user_message,
        stop_reason=request.stop_reason
    )
    
    # Get current state
    config_data = session.get("config", {})
    from app.voice_interview.schemas import VoiceInterviewConfigRequest
    config = VoiceInterviewConfigRequest(**config_data)
    
    current_phase = session.get("current_phase", "greeting")
    conversation_history = session_manager.get_conversation_history(request.session_id)
    current_question = session_manager.get_current_question(request.session_id)
    
    # Generate appropriate response based on phase
    if current_phase == "greeting":
        # After greeting, move to first question
        response_text = conversation_manager.generate_response(
            config, request.user_message, conversation_history, current_phase, current_question
        )
        
        session_manager.set_phase(request.session_id, "questions")
        
        first_question = session_manager.get_current_question(request.session_id)
        if first_question:
            question_text = conversation_manager.generate_next_question(
                config, first_question, conversation_history, is_first_question=True
            )
            response_text = f"{response_text} {question_text}"
    
    elif current_phase == "questions":
        # User answered a question
        can_followup = session_manager.can_ask_followup(request.session_id)
        should_ask_followup = False
        
        if can_followup:
            followup = conversation_manager.generate_followup(
                config, current_question, request.user_message, conversation_history
            )
            
            if followup:
                response_text = followup
                session_manager.increment_followup(request.session_id)
                should_ask_followup = True
            else:
                # No follow-up needed, acknowledge and move to next question
                response_text = conversation_manager.generate_response(
                    config, request.user_message, conversation_history, current_phase, current_question
                )
                session_manager.advance_to_next_question(request.session_id)
                next_question = session_manager.get_current_question(request.session_id)
                
                if next_question:
                    question_text = conversation_manager.generate_next_question(
                        config, next_question, conversation_history, is_first_question=False
                    )
                    response_text = f"{response_text} {question_text}"
                else:
                    # No more questions, move to wrapup
                    session_manager.set_phase(request.session_id, "wrapup")
                    wrapup_text = conversation_manager.generate_wrapup(config, conversation_history)
                    response_text = f"{response_text} {wrapup_text}"
        else:
            # Max follow-ups reached, move to next question
            response_text = conversation_manager.generate_response(
                config, request.user_message, conversation_history, current_phase, current_question
            )
            session_manager.advance_to_next_question(request.session_id)
            next_question = session_manager.get_current_question(request.session_id)
            
            if next_question:
                question_text = conversation_manager.generate_next_question(
                    config, next_question, conversation_history, is_first_question=False
                )
                response_text = f"{response_text} {question_text}"
            else:
                # No more questions, move to wrapup
                session_manager.set_phase(request.session_id, "wrapup")
                wrapup_text = conversation_manager.generate_wrapup(config, conversation_history)
                response_text = f"{response_text} {wrapup_text}"
    
    elif current_phase == "followup":
        # User responded to follow-up
        response_text = conversation_manager.generate_response(
            config, request.user_message, conversation_history, current_phase, current_question
        )
        session_manager.advance_to_next_question(request.session_id)
        next_question = session_manager.get_current_question(request.session_id)
        
        if next_question:
            question_text = conversation_manager.generate_next_question(
                config, next_question, conversation_history, is_first_question=False
            )
            response_text = f"{response_text} {question_text}"
        else:
            # No more questions, move to wrapup
            session_manager.set_phase(request.session_id, "wrapup")
            wrapup_text = conversation_manager.generate_wrapup(config, conversation_history)
            response_text = f"{response_text} {wrapup_text}"
    
    elif current_phase == "wrapup":
        # Interview is wrapping up
        response_text = "Thank you for your time. The interview is now complete."
        session_manager.set_phase(request.session_id, "complete")
    
    else:
        # Default response
        response_text = conversation_manager.generate_response(
            config, request.user_message, conversation_history, current_phase, current_question
        )
    
    # Add AI turn to conversation
    session_manager.add_conversation_turn(request.session_id, "ai", response_text)
    
    # Generate audio if requested (async)
    audio_base64 = None
    if request.include_audio:
        audio_base64 = await text_to_speech.synthesize_to_base64_async(
            text=response_text,
            language_code="en-US",
            voice_name=None,  # Uses default voice based on gender
            ssml_gender="NEUTRAL",
            audio_format="mp3"
        )
    
    # Get updated state
    state = session_manager.get_state(request.session_id)
    
    return AIResponse(
        session_id=request.session_id,
        text=response_text,
        audio_data=audio_base64,
        audio_format="mp3",
        current_phase=state.current_phase if state else current_phase,
        current_question_index=state.current_question_index if state else 0,
        total_questions=state.total_questions if state else 0,
        progress_percentage=state.progress_percentage if state else 0.0,
        is_complete=state.is_complete if state else False,
        should_ask_followup=should_ask_followup if 'should_ask_followup' in locals() else False
    )


@router.get("/{session_id}/state", response_model=InterviewState)
async def get_interview_state(session_id: str):
    """
    Get current state of a voice interview session.
    
    Node.js gateway can call this to check session status.
    """
    state = session_manager.get_state(session_id)
    
    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    
    return state


@router.get("/{session_id}/analysis", response_model=VoiceInterviewAnalysis)
async def get_analysis(session_id: str):
    """
    Get comprehensive analysis of the completed voice interview.
    
    Node.js gateway should call this when interview is complete.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    
    conversation_turns = session_manager.get_conversation_history(session_id)
    config = session.get("config", {})
    
    try:
        analysis = analyzer.generate_analysis(
            session_id=session_id,
            conversation_turns=conversation_turns,
            config=config
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
    Get current status of a voice interview session.
    
    Returns basic information about the session state.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    
    state = session_manager.get_state(session_id)
    is_complete = session_manager.is_complete(session_id)
    conversation_turns = session_manager.get_conversation_history(session_id)
    
    return {
        "session_id": session_id,
        "is_complete": is_complete,
        "is_started": session.get("is_started", False),
        "current_phase": session.get("current_phase", "greeting"),
        "current_question_index": session.get("current_question_index", 0),
        "total_questions": len(session.get("questions", [])),
        "conversation_turns": len(conversation_turns),
        "created_at": session.get("created_at"),
        "started_at": session.get("started_at"),
        "completed_at": session.get("completed_at")
    }

