# Voice Interview Feature - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [User Flow](#user-flow)
4. [API Endpoints](#api-endpoints)
5. [Internal Components](#internal-components)
6. [Data Flow](#data-flow)
7. [State Management](#state-management)
8. [Interview Phases](#interview-phases)
9. [Error Handling](#error-handling)
10. [Configuration](#configuration)
11. [Integration Points](#integration-points)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The Voice Interview feature is a comprehensive AI-powered interview simulation system that allows users to practice interviews through voice interaction. The system uses:

- **Google Gemini** for AI conversation generation
- **Eleven Labs** for Speech-to-Text (STT) and Text-to-Speech (TTS)
- **Redis** for session state management
- **FastAPI** for REST API endpoints

### Key Features
- Real-time voice transcription
- AI-generated interview questions based on job role and experience level
- Natural conversation flow with follow-up questions
- Comprehensive performance analysis after completion
- Multiple interviewer personas (HR, Technical, Behavioral, Mixed)
- Session persistence with Redis

---

## Architecture

### System Components

```
┌─────────────────┐
│  Client (Web)   │
│  (Node.js UI)   │
└────────┬────────┘
         │
         │ HTTP/REST
         │
┌────────▼─────────────────────────────────────────┐
│         FastAPI Backend                          │
│  ┌──────────────────────────────────────────┐   │
│  │  Router (API Endpoints)                  │   │
│  └──────┬───────────────────────────────────┘   │
│         │                                        │
│  ┌──────▼──────────┐  ┌──────────────────────┐ │
│  │ Session Manager │  │ Conversation Manager │ │
│  │  (Redis)        │  │  (Gemini LLM)        │ │
│  └──────┬──────────┘  └──────┬───────────────┘ │
│         │                     │                  │
│  ┌──────▼──────────┐  ┌──────▼───────────────┐ │
│  │ Question        │  │ Speech Services      │ │
│  │ Generator       │  │ (Eleven Labs)        │ │
│  │ (Gemini LLM)    │  │ - STT                │ │
│  └─────────────────┘  │ - TTS                │ │
│                       └──────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  Analyzer (Post-Interview Analysis)      │ │
│  │  (Gemini LLM)                            │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
         │                    │
         │                    │
    ┌────▼────┐         ┌─────▼─────┐
    │  Redis  │         │  Gemini   │
    │  Cloud  │         │   API     │
    └─────────┘         └───────────┘
```

### Technology Stack
- **Backend Framework**: FastAPI (Python)
- **LLM**: Google Gemini 2.5 Flash
- **STT/TTS**: Eleven Labs API
- **Session Storage**: Redis (with in-memory fallback)
- **Question Generation**: Custom prompt engineering with Gemini

---

## User Flow

### Complete Interview Journey

#### Phase 1: Session Creation
1. **User initiates interview**
   - User fills out interview configuration form:
     - Job role (e.g., "Software Developer Intern")
     - Experience level (Entry/Mid/Senior/Executive)
     - Company (optional)
     - Job description (optional)
     - Interview type (Technical/Behavioral/System Design/Mixed)
     - Number of questions (3-15, default: 5)
     - Interviewer role (HR/Technical/Behavioral/Mixed)

2. **Client calls `/voice-interview/create`**
   - Sends `VoiceInterviewConfigRequest` with all configuration
   - Backend generates questions using `InterviewQuestionGenerator`
   - Creates session in Redis with:
     - Unique `session_id` (UUID)
     - Generated questions
     - Configuration
     - Initial state (phase: "greeting", question_index: 0)
   - Returns `VoiceInterviewSessionResponse` with `session_id`

3. **Client receives session_id**
   - Stores `session_id` for subsequent API calls
   - Displays "Ready to start" UI

#### Phase 2: Interview Start
4. **User clicks "Start Interview"**
   - Client calls `/voice-interview/start` with `session_id`
   - Backend:
     - Marks session as started (`is_started: true`)
     - Generates personalized greeting using `VoiceConversationManager`
     - Adds greeting to conversation history
     - Generates audio using Eleven Labs TTS
     - Returns `AIResponse` with:
       - Greeting text
       - Base64-encoded audio (MP3)
       - Current phase: "greeting"
       - Progress: 0%

5. **Client plays greeting audio**
   - Displays greeting text in chat UI
   - Plays audio to user
   - Shows "LISTENING" indicator
   - Waits for user response

#### Phase 3: Question & Answer Loop
6. **User speaks answer**
   - Client captures audio from microphone
   - Audio is recorded in chunks (WebM format)
   - Client sends audio to `/voice-interview/transcribe`
   - Backend:
     - Decodes base64 audio
     - Sends to Eleven Labs STT API
     - Returns transcription with confidence score

7. **Client processes transcription**
   - If transcription successful:
     - Displays transcribed text in chat
     - Calls `/voice-interview/process` with transcribed text
   - If transcription fails:
     - Shows error message
     - Allows user to retry

8. **Backend processes user message**
   - Adds user turn to conversation history
   - Determines current phase and generates appropriate response:
   
   **If phase = "greeting":**
     - Generates acknowledgment response
     - Transitions to "questions" phase
     - Gets first question
     - Rephrases question naturally
     - Combines: acknowledgment + first question
   
   **If phase = "questions":**
     - Checks if follow-up allowed (max 2 per question)
     - If follow-up allowed:
       - Analyzes answer length and quality
       - Generates follow-up question if needed
       - If follow-up generated:
         - Sets phase to "followup"
         - Increments followup_count
         - Returns follow-up question
       - If no follow-up needed:
         - Generates acknowledgment
         - Advances to next question
         - Returns acknowledgment + next question
     - If max follow-ups reached:
       - Generates acknowledgment
       - Advances to next question
       - Returns acknowledgment + next question
   
   **If phase = "followup":**
     - Generates acknowledgment
     - Advances to next question
     - Returns acknowledgment + next question
   
   **If no more questions:**
     - Transitions to "wrapup" phase
     - Generates wrap-up message
     - Returns wrap-up

9. **Client receives AI response**
   - Displays AI response text in chat
   - If `include_audio=true`:
     - Receives base64-encoded audio
     - Plays audio to user
   - Updates progress bar
   - Updates question counter
   - If `is_complete=true`:
     - Shows completion message
     - Transitions to analysis phase

10. **Repeat steps 6-9** until all questions answered

#### Phase 4: Wrap-up
11. **Interview completion**
    - When last question answered:
      - Backend generates wrap-up message
      - Sets phase to "wrapup"
      - Marks session as complete
    - Client displays wrap-up message
    - Shows "Interview Complete" UI

#### Phase 5: Analysis
12. **Generate analysis**
    - Client calls `/voice-interview/{session_id}/analysis`
    - Backend:
      - Retrieves full conversation history
      - Sends to `VoiceInterviewAnalyzer`
      - Analyzer uses Gemini to evaluate:
        - Communication score (0-100)
        - Content score (0-100)
        - Engagement score (0-100)
        - Overall score (average)
        - Strengths (list)
        - Improvement areas (list)
        - Recommendations (list)
        - Conversation summary
      - Returns `VoiceInterviewAnalysis`

13. **Display results**
    - Client shows analysis dashboard:
      - Overall score
      - Breakdown by category
      - Strengths and improvements
      - Recommendations
      - Full conversation transcript

---

## API Endpoints

### 1. POST `/voice-interview/create`
**Purpose**: Create a new interview session with generated questions

**Request Body**:
```json
{
  "job_role": "Software Developer Intern",
  "experience_level": "Entry Level (0-2 years)",
  "company": "Tech Corp",
  "job_description": "Optional job description...",
  "interview_type": "Technical",
  "interview_role": "Technical Interviewer",
  "difficulty": "Intermediate",
  "user_id": "user123",
  "num_questions": 5,
  "duration_minutes": 30
}
```

**Response** (201 Created):
```json
{
  "session_id": "uuid-here",
  "total_questions": 5,
  "config": { /* full config */ },
  "created_at": "2024-01-01T12:00:00Z"
}
```

**Process**:
1. Converts config to `InterviewConfigRequest`
2. Calls `question_generator.generate_questions()`
3. Creates session in Redis
4. Returns session response

**Error Cases**:
- 500: Question generation failed
- 500: Session creation failed

---

### 2. POST `/voice-interview/start`
**Purpose**: Start the interview and get greeting

**Request Body**:
```json
{
  "session_id": "uuid-here"
}
```

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "text": "Hello! Thank you for joining...",
  "audio_data": "base64-encoded-mp3",
  "audio_format": "mp3",
  "current_phase": "greeting",
  "current_question_index": 0,
  "total_questions": 5,
  "progress_percentage": 0.0,
  "is_complete": false,
  "should_ask_followup": false
}
```

**Process**:
1. Validates session exists
2. Marks session as started
3. Generates greeting via `conversation_manager.generate_greeting()`
4. Adds greeting to conversation history
5. Generates audio via Eleven Labs TTS
6. Returns response with audio

**Error Cases**:
- 404: Session not found

---

### 3. POST `/voice-interview/transcribe`
**Purpose**: Transcribe audio chunk to text

**Request Body**:
```json
{
  "session_id": "uuid-here",
  "audio_data": "base64-encoded-audio",
  "audio_format": "webm",
  "sample_rate": 16000
}
```

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "text": "I have experience with Python...",
  "confidence": 0.9,
  "is_final": true
}
```

**Process**:
1. Decodes base64 audio
2. Sends to Eleven Labs STT API
3. Returns transcription and confidence

**Error Cases**:
- 500: Transcription failed
- 500: Invalid audio format

---

### 4. POST `/voice-interview/process`
**Purpose**: Process user message and generate AI response

**Request Body**:
```json
{
  "session_id": "uuid-here",
  "user_message": "I have experience with Python and JavaScript...",
  "include_audio": true
}
```

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "text": "That's great! Let's move on to the next question. Can you explain...",
  "audio_data": "base64-encoded-mp3",
  "audio_format": "mp3",
  "current_phase": "questions",
  "current_question_index": 1,
  "total_questions": 5,
  "progress_percentage": 20.0,
  "is_complete": false,
  "should_ask_followup": false
}
```

**Process**:
1. Validates session exists
2. Adds user message to conversation history
3. Gets current phase and question
4. Generates response based on phase (see [Interview Phases](#interview-phases))
5. Adds AI response to conversation history
6. Generates audio if requested
7. Returns response with updated state

**Error Cases**:
- 404: Session not found
- 500: Response generation failed

---

### 5. GET `/voice-interview/{session_id}/state`
**Purpose**: Get current interview state

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "current_phase": "questions",
  "current_question_index": 2,
  "total_questions": 5,
  "progress_percentage": 40.0,
  "is_complete": false,
  "is_started": true
}
```

**Error Cases**:
- 404: Session not found

---

### 6. GET `/voice-interview/{session_id}/analysis`
**Purpose**: Get comprehensive interview analysis

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "overall_score": 78.5,
  "total_questions": 5,
  "answered_questions": 5,
  "conversation_summary": "The candidate demonstrated...",
  "strengths_summary": [
    "Clear communication",
    "Strong technical knowledge"
  ],
  "improvement_areas": [
    "Could provide more specific examples",
    "Pace could be slower"
  ],
  "recommendations": [
    "Practice using STAR method",
    "Prepare more concrete examples"
  ],
  "communication_score": 82.0,
  "content_score": 75.0,
  "engagement_score": 78.5,
  "completed_at": "2024-01-01T12:30:00Z"
}
```

**Process**:
1. Retrieves conversation history
2. Sends to `VoiceInterviewAnalyzer.generate_analysis()`
3. Analyzer uses Gemini to evaluate performance
4. Returns comprehensive analysis

**Error Cases**:
- 404: Session not found
- 500: Analysis generation failed

---

### 7. GET `/voice-interview/{session_id}/status`
**Purpose**: Get basic session status

**Response** (200 OK):
```json
{
  "session_id": "uuid-here",
  "is_complete": false,
  "is_started": true,
  "current_phase": "questions",
  "current_question_index": 2,
  "total_questions": 5,
  "conversation_turns": 8,
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:01:00Z",
  "completed_at": null
}
```

---

## Internal Components

### 1. VoiceInterviewSessionManager
**Location**: `app/voice_interview/session_manager.py`

**Responsibilities**:
- Session lifecycle management (create, read, update)
- Redis integration with in-memory fallback
- Conversation history tracking
- State management (phase, question index, progress)
- Follow-up counter management

**Key Methods**:
- `create_session()`: Creates new session in Redis
- `get_session()`: Retrieves session data
- `add_conversation_turn()`: Adds user/AI message to history
- `get_conversation_history()`: Returns all conversation turns
- `get_current_question()`: Gets current question object
- `advance_to_next_question()`: Moves to next question
- `increment_followup()`: Increments follow-up counter
- `can_ask_followup()`: Checks if follow-up allowed
- `set_phase()`: Updates interview phase
- `get_state()`: Returns current `InterviewState`

**Session Data Structure**:
```python
{
  "session_id": "uuid",
  "config": { /* VoiceInterviewConfigRequest */ },
  "questions": [ /* List of InterviewQuestion */ ],
  "conversation_turns": [
    {
      "turn_id": 0,
      "speaker": "ai" | "user",
      "text": "message text",
      "timestamp": "ISO datetime"
    }
  ],
  "current_phase": "greeting" | "questions" | "followup" | "wrapup" | "complete",
  "current_question_index": 0,
  "followup_count": 0,
  "max_followups_per_question": 2,
  "created_at": "ISO datetime",
  "started_at": "ISO datetime" | null,
  "completed_at": "ISO datetime" | null,
  "is_complete": false,
  "is_started": false
}
```

**Redis Key Format**: `voice_interview:session:{session_id}`
**TTL**: 86400 seconds (24 hours)

---

### 2. VoiceConversationManager
**Location**: `app/voice_interview/conversation_manager.py`

**Responsibilities**:
- Generate natural conversation responses
- Manage interview flow (greeting, questions, follow-ups, wrap-up)
- Adapt to interviewer role persona
- Use Gemini LLM for text generation

**Key Methods**:
- `generate_greeting()`: Creates personalized greeting
- `generate_next_question()`: Rephrases question naturally
- `generate_followup()`: Generates follow-up based on answer
- `generate_response()`: General response generation
- `generate_wrapup()`: Creates conclusion message

**Persona Mapping**:
- `HR_RECRUITER`: "friendly HR recruiter focused on cultural fit and soft skills"
- `TECHNICAL_INTERVIEWER`: "technical interviewer focused on problem-solving and technical skills"
- `BEHAVIORAL_INTERVIEWER`: "behavioral interviewer focused on past experiences and soft skills"
- `MIXED_INTERVIEWER`: "professional interviewer covering both technical and behavioral aspects"

**Follow-up Logic**:
- If answer < 20 words: Always asks follow-up ("Could you tell me a bit more?")
- If answer > 300 words: No follow-up (too long)
- Otherwise: LLM decides based on answer quality

---

### 3. InterviewQuestionGenerator
**Location**: `app/text_interview/generator.py`

**Responsibilities**:
- Generate interview questions based on configuration
- Use Gemini LLM with structured prompts
- Parse JSON responses with fallback handling
- Create `InterviewQuestion` objects

**Question Generation Process**:
1. Builds detailed prompt with:
   - Job role and experience level
   - Interview type guidance
   - Experience level expectations
   - JSON format requirements
2. Calls Gemini API
3. Parses JSON response (with retry on parse failure)
4. Converts to `InterviewQuestion` objects
5. Falls back to default questions if generation fails

**Retry Logic**:
- Only retries on JSON parsing errors (max 1 retry)
- Does NOT retry on Gemini blocking or other errors
- Falls back to heuristic parser if JSON fails
- Falls back to default questions if all parsing fails

---

### 4. SpeechToTextService
**Location**: `app/voice_interview/speech_to_text.py`

**Responsibilities**:
- Convert audio to text using Eleven Labs STT
- Handle base64 encoding/decoding
- Language code conversion (ISO 639-3)

**Key Methods**:
- `transcribe_audio()`: Transcribe raw audio bytes
- `transcribe_base64()`: Transcribe base64-encoded audio

**Eleven Labs Configuration**:
- Model: `scribe_v2`
- Language: Auto-detect or ISO 639-3 code
- Format: webm, wav, flac, mp3 supported
- Confidence: Default 0.9 (Eleven Labs doesn't provide confidence)

---

### 5. TextToSpeechService
**Location**: `app/voice_interview/speech_to_text.py`

**Responsibilities**:
- Convert text to speech using Eleven Labs TTS
- Generate base64-encoded audio
- Voice selection based on gender preference

**Key Methods**:
- `synthesize_speech()`: Generate audio bytes
- `synthesize_to_base64()`: Generate base64 audio string

**Eleven Labs Configuration**:
- Model: `eleven_multilingual_v2`
- Default Voice (Female): `21m00Tcm4TlvDq8ikWAM` (Rachel)
- Default Voice (Male): `EXAVITQu4vr4xnSDxMaL` (Bella)
- Format: MP3 (recommended)
- Voice Settings:
  - Stability: 0.5
  - Similarity Boost: 0.75
  - Style: 0.0
  - Speaker Boost: true

---

### 6. VoiceInterviewAnalyzer
**Location**: `app/voice_interview/analyzer.py`

**Responsibilities**:
- Analyze completed interviews
- Generate performance scores
- Provide feedback and recommendations
- Use Gemini LLM for evaluation

**Analysis Dimensions**:
1. **Communication Score (0-100)**: Clarity, articulation, pace, tone
2. **Content Score (0-100)**: Relevance, depth, examples, accuracy
3. **Engagement Score (0-100)**: Enthusiasm, confidence, responsiveness

**Analysis Output**:
- Overall score (average of three dimensions)
- Conversation summary (3-4 sentences)
- Top 3-5 strengths
- Top 3-5 improvement areas
- 3-5 actionable recommendations

**Fallback Analysis**:
- If LLM fails, uses heuristic analysis:
  - Communication: Based on average words per turn
  - Content: Based on total word count
  - Engagement: Default 70
  - Basic strengths/improvements based on metrics

---

## Data Flow

### Question Generation Flow
```
Client Request
    ↓
POST /voice-interview/create
    ↓
InterviewQuestionGenerator.generate_questions()
    ↓
Gemini API (generate_raw)
    ↓
Parse JSON Response
    ↓
Create InterviewQuestion Objects
    ↓
VoiceInterviewSessionManager.create_session()
    ↓
Store in Redis
    ↓
Return session_id
```

### Conversation Flow
```
User Speaks
    ↓
Client Captures Audio
    ↓
POST /voice-interview/transcribe
    ↓
SpeechToTextService.transcribe_base64()
    ↓
Eleven Labs STT API
    ↓
Return Transcription
    ↓
Client Calls POST /voice-interview/process
    ↓
VoiceConversationManager.generate_response()
    ↓
Gemini API (generate_raw)
    ↓
TextToSpeechService.synthesize_to_base64()
    ↓
Eleven Labs TTS API
    ↓
Return AI Response + Audio
    ↓
Client Plays Audio
```

### Analysis Flow
```
Interview Complete
    ↓
GET /voice-interview/{session_id}/analysis
    ↓
VoiceInterviewAnalyzer.generate_analysis()
    ↓
Build Conversation Text
    ↓
Gemini API (generate_raw)
    ↓
Parse Analysis JSON
    ↓
Create VoiceInterviewAnalysis
    ↓
Return Analysis
```

---

## State Management

### Interview Phases

1. **greeting**
   - Initial phase when interview starts
   - AI delivers greeting
   - Transitions to "questions" after user responds

2. **questions**
   - Main interview phase
   - AI asks questions from generated list
   - User provides answers
   - Can transition to "followup" if follow-up needed
   - Transitions to "wrapup" when all questions answered

3. **followup**
   - Temporary phase during follow-up questions
   - Max 2 follow-ups per question
   - Transitions back to "questions" after follow-up answered

4. **wrapup**
   - Final phase when interview concludes
   - AI delivers wrap-up message
   - Transitions to "complete"

5. **complete**
   - Interview is finished
   - No further conversation
   - Ready for analysis

### State Transitions
```
greeting → questions → followup → questions → ... → wrapup → complete
                ↑_________________|
```

### Progress Calculation
```python
progress_percentage = (current_question_index + 1) / total_questions * 100
```

### Follow-up Logic
- Max 2 follow-ups per question
- Follow-up triggered if:
  - Answer too short (< 20 words)
  - LLM determines follow-up needed
- Follow-up NOT triggered if:
  - Answer too long (> 300 words)
  - Max follow-ups already reached
  - LLM returns "NONE"

---

## Interview Phases

### Phase: greeting
**Trigger**: User clicks "Start Interview"

**Actions**:
1. Generate personalized greeting
2. Add greeting to conversation history
3. Generate audio
4. Return greeting + audio

**User Response Handling**:
- User responds to greeting
- System acknowledges
- Transitions to "questions" phase
- Delivers first question

---

### Phase: questions
**Trigger**: After greeting or after follow-up answered

**Actions**:
1. Get current question from list
2. Rephrase question naturally
3. Deliver question to user
4. Wait for user answer

**User Response Handling**:
- Check if follow-up allowed
- If yes:
  - Analyze answer
  - Generate follow-up if needed
  - If follow-up: transition to "followup"
  - If no follow-up: advance to next question
- If no (max follow-ups):
  - Advance to next question
- If no more questions:
  - Transition to "wrapup"

---

### Phase: followup
**Trigger**: Follow-up question generated

**Actions**:
1. Deliver follow-up question
2. Wait for user answer

**User Response Handling**:
- Acknowledge answer
- Advance to next question
- Transition back to "questions"
- If no more questions: transition to "wrapup"

---

### Phase: wrapup
**Trigger**: All questions answered

**Actions**:
1. Generate wrap-up message
2. Add to conversation history
3. Mark session as complete
4. Return wrap-up

**User Response Handling**:
- Interview is complete
- No further processing
- Ready for analysis

---

## Error Handling

### Question Generation Errors
- **Gemini API Failure**: Falls back to default questions
- **JSON Parse Error**: Retries once, then uses fallback parser
- **Empty Response**: Uses default questions

### Transcription Errors
- **Eleven Labs STT Failure**: Returns empty string with confidence 0.0
- **Invalid Audio Format**: Returns error response
- **Network Timeout**: Returns error response

### Conversation Errors
- **Gemini API Failure**: Uses default responses
- **Empty Response**: Uses fallback text
- **Session Not Found**: Returns 404 error

### Analysis Errors
- **LLM Failure**: Uses heuristic fallback analysis
- **JSON Parse Error**: Uses fallback analysis
- **Empty Conversation**: Returns empty analysis

### Session Errors
- **Redis Connection Failure**: Falls back to in-memory storage
- **Session Expired**: Returns 404 error
- **Invalid Session ID**: Returns 404 error

---

## Configuration

### Environment Variables
```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Eleven Labs API
ELEVEN_LABS_API_KEY=your_eleven_labs_api_key

# Redis
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
```

### Default Settings
- **Max Follow-ups per Question**: 2
- **Session TTL**: 24 hours
- **Default Voice**: Rachel (Female, Professional)
- **Audio Format**: MP3
- **Sample Rate**: 16000 Hz
- **Default Questions**: 5
- **Min Questions**: 3
- **Max Questions**: 15

### Interviewer Personas
- **HR Recruiter**: Focus on cultural fit, soft skills
- **Technical Interviewer**: Focus on problem-solving, technical skills
- **Behavioral Interviewer**: Focus on past experiences, soft skills
- **Mixed Interviewer**: Balanced approach

---

## Integration Points

### Client Integration (Node.js Gateway)
The client should:
1. Handle audio recording (WebRTC/MediaRecorder)
2. Convert audio to base64
3. Call transcription endpoint
4. Display transcribed text
5. Call process endpoint
6. Play audio responses
7. Update UI state based on response
8. Poll state endpoint if needed
9. Call analysis endpoint on completion

### Expected Client Flow
```javascript
// 1. Create session
const session = await fetch('/voice-interview/create', {
  method: 'POST',
  body: JSON.stringify(config)
});

// 2. Start interview
const greeting = await fetch('/voice-interview/start', {
  method: 'POST',
  body: JSON.stringify({ session_id })
});

// 3. Record audio and transcribe
const transcript = await fetch('/voice-interview/transcribe', {
  method: 'POST',
  body: JSON.stringify({
    session_id,
    audio_data: base64Audio,
    audio_format: 'webm'
  })
});

// 4. Process message
const response = await fetch('/voice-interview/process', {
  method: 'POST',
  body: JSON.stringify({
    session_id,
    user_message: transcript.text,
    include_audio: true
  })
});

// 5. Play audio and repeat steps 3-4

// 6. Get analysis
const analysis = await fetch(`/voice-interview/${session_id}/analysis`);
```

---

## Troubleshooting

### Issue: "Processing your answer..." stuck
**Possible Causes**:
1. Transcription endpoint not called or failed
2. Process endpoint waiting for Gemini response
3. Empty user message causing LLM to hang
4. Network timeout

**Solutions**:
- Check browser network tab for pending requests
- Verify audio is being captured
- Check server logs for errors
- Ensure user message is not empty
- Add timeout handling in client

### Issue: Questions not generating
**Possible Causes**:
1. Gemini API key invalid
2. Gemini blocking output
3. JSON parsing failing repeatedly

**Solutions**:
- Verify GEMINI_API_KEY is set
- Check server logs for Gemini errors
- Review question generation retry logic
- Check if fallback questions are used

### Issue: Audio not playing
**Possible Causes**:
1. TTS API failure
2. Invalid audio format
3. Base64 encoding issue

**Solutions**:
- Verify ELEVEN_LABS_API_KEY is set
- Check audio format compatibility
- Verify base64 decoding in client
- Test audio playback separately

### Issue: Session not found
**Possible Causes**:
1. Session expired (24 hour TTL)
2. Redis connection lost
3. Invalid session_id

**Solutions**:
- Check Redis connection
- Verify session_id format
- Check session TTL
- Review in-memory fallback

### Issue: Follow-ups not working
**Possible Causes**:
1. Follow-up counter not incrementing
2. Max follow-ups reached
3. LLM returning "NONE"

**Solutions**:
- Check followup_count in session
- Verify max_followups_per_question setting
- Review follow-up generation logic
- Check answer length thresholds

---

## Best Practices

### For Developers
1. **Always handle errors gracefully**: Use fallbacks for LLM failures
2. **Validate session state**: Check session exists before operations
3. **Log important events**: Track conversation flow for debugging
4. **Handle timeouts**: Set appropriate timeouts for API calls
5. **Test edge cases**: Empty responses, long answers, network failures

### For Client Integration
1. **Implement retry logic**: For transient failures
2. **Show loading states**: During transcription and processing
3. **Handle audio errors**: Graceful degradation if audio fails
4. **Poll state if needed**: For long-running operations
5. **Cache session_id**: Store for reconnection

### For Production
1. **Monitor API usage**: Track Gemini and Eleven Labs quotas
2. **Set up alerts**: For high error rates
3. **Optimize prompts**: Reduce token usage
4. **Cache common responses**: Reduce LLM calls
5. **Scale Redis**: For high concurrent sessions

---

## Future Enhancements

### Potential Improvements
1. **Real-time streaming**: WebSocket for live transcription
2. **Multiple languages**: Support for non-English interviews
3. **Custom voices**: User-selectable TTS voices
4. **Interview templates**: Pre-defined question sets
5. **Video support**: Add video interview capability
6. **Analytics dashboard**: Track interview performance over time
7. **Export transcripts**: PDF/Word export of conversations
8. **Practice mode**: Unlimited retries on questions
9. **Peer review**: Share interviews for feedback
10. **Integration with ATS**: Connect with applicant tracking systems

---

## Conclusion

The Voice Interview feature provides a comprehensive, AI-powered interview simulation system. By understanding the architecture, data flow, and integration points, developers can effectively use, extend, and troubleshoot the system.

For questions or issues, refer to:
- Server logs for detailed error messages
- API response status codes
- This documentation for architecture details
- Code comments for implementation specifics

