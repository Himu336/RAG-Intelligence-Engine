# Voice Interview API Reference

This document describes the REST API endpoints for the Voice Interview AI microservice. The Node.js gateway should call these endpoints to handle voice interview functionality.

## Base URL
All endpoints are prefixed with `/voice-interview`

## Endpoints

### 1. Create Interview Session
**POST** `/voice-interview/create`

Creates a new voice interview session with generated questions.

**Request Body:**
```json
{
  "job_role": "Senior Software Engineer",
  "experience_level": "Senior Level (6-10 years)",
  "company": "Google",
  "job_description": "Optional job description...",
  "interview_type": "Technical",
  "interview_role": "Technical Interviewer",
  "difficulty": "Advanced",
  "user_id": "user123",
  "num_questions": 5,
  "duration_minutes": 30
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "total_questions": 5,
  "config": { ... },
  "created_at": "2024-01-01T00:00:00"
}
```

---

### 2. Transcribe Audio
**POST** `/voice-interview/transcribe`

Transcribes audio chunk to text. Call this when receiving audio from the client.

**Request Body:**
```json
{
  "session_id": "uuid-here",
  "audio_data": "base64-encoded-audio",
  "audio_format": "webm",
  "sample_rate": 16000
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "text": "Transcribed text here",
  "confidence": 0.95,
  "is_final": true
}
```

---

### 3. Start Interview
**POST** `/voice-interview/start`

Starts the interview and returns the greeting. Call this when user clicks "Start Interview".

**Request Body:**
```json
{
  "session_id": "uuid-here"
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "text": "Hello! Thank you for joining us today...",
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

---

### 4. Process Message
**POST** `/voice-interview/process`

Main endpoint for conversation flow. Call this after transcribing user audio or receiving text input.

**Request Body:**
```json
{
  "session_id": "uuid-here",
  "user_message": "User's transcribed text or input",
  "include_audio": true
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "text": "AI response text",
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

**Flow:**
- If `current_phase` is "greeting" → moves to first question
- If `current_phase` is "questions" → processes answer, may ask follow-up or move to next question
- If `current_phase` is "followup" → processes follow-up response, moves to next question
- If `current_phase` is "wrapup" → concludes interview
- If `is_complete` is true → interview is finished

---

### 5. Get Interview State
**GET** `/voice-interview/{session_id}/state`

Get current state of the interview session.

**Response:**
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

---

### 6. Get Session Status
**GET** `/voice-interview/{session_id}/status`

Get basic status information about the session.

**Response:**
```json
{
  "session_id": "uuid-here",
  "is_complete": false,
  "is_started": true,
  "current_phase": "questions",
  "current_question_index": 2,
  "total_questions": 5,
  "conversation_turns": 6,
  "created_at": "2024-01-01T00:00:00",
  "started_at": "2024-01-01T00:05:00",
  "completed_at": null
}
```

---

### 7. Get Analysis
**GET** `/voice-interview/{session_id}/analysis`

Get comprehensive analysis of the completed interview. Call this when interview is finished.

**Response:**
```json
{
  "session_id": "uuid-here",
  "overall_score": 75.5,
  "total_questions": 5,
  "answered_questions": 5,
  "conversation_summary": "Summary of the interview...",
  "strengths_summary": ["Clear communication", "Good examples"],
  "improvement_areas": ["Could be more concise", "Add metrics"],
  "recommendations": ["Practice speaking pace", "Prepare STAR examples"],
  "communication_score": 80.0,
  "content_score": 72.0,
  "engagement_score": 74.5,
  "completed_at": "2024-01-01T01:00:00"
}
```

---

## Interview Flow

1. **Create Session**: Node.js gateway calls `/create` when user wants to start an interview
2. **Start Interview**: When user clicks "Start", call `/start` to get greeting
3. **Audio Processing Loop**:
   - Receive audio from client via WebSocket
   - Call `/transcribe` to convert audio to text
   - Call `/process` with transcribed text to get AI response
   - Send AI response (text + audio) back to client via WebSocket
4. **Continue**: Repeat step 3 until `is_complete` is true
5. **Get Analysis**: Call `/analysis` when interview is complete

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Session not found
- `500`: Internal server error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Notes

- All audio is base64-encoded
- Audio format defaults: input `webm`, output `mp3`
- Session TTL: 24 hours
- Maximum follow-ups per question: 2
- The service handles conversation state internally
- Node.js gateway should manage WebSocket connections and call these REST endpoints

