# Interview API Reference

This document describes the Interview API endpoints for integration with your Node.js server.

## Base URL
All endpoints are prefixed with `/interview`

## Endpoints

### 1. Create Interview Session
**POST** `/interview/create`

Creates a new interview session with AI-generated questions.

**Request Body:**
```json
{
  "job_role": "Senior Software Engineer",
  "experience_level": "Mid Level (3-5 years)",
  "company": "Google",  // optional
  "job_description": "Full job description text...",  // optional
  "interview_type": "Behavioral",
  "user_id": "user123",
  "num_questions": 5  // default: 5, range: 3-15
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

### 2. Get Current Question
**GET** `/interview/{session_id}/question`

Retrieves the current question for the interview session.

**Response:**
```json
{
  "session_id": "uuid-here",
  "current_question": {
    "question_id": 0,
    "question_text": "Tell me about yourself...",
    "difficulty": "Easy",
    "hint": {
      "text": "Use the present-past-future framework...",
      "framework": "Present-Past-Future"
    },
    "interview_type": "Behavioral"
  },
  "question_number": 1,
  "total_questions": 5,
  "progress_percentage": 20.0,
  "is_complete": false
}
```

---

### 3. Submit Answer
**POST** `/interview/{session_id}/answer`

Submits an answer for the current question and advances to the next.

**Request Body:**
```json
{
  "answer": "User's answer text here...",
  "question_id": 0
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "question_id": 0,
  "answer_saved": true,
  "next_question_available": true,
  "is_complete": false
}
```

---

### 4. Get Analysis
**GET** `/interview/{session_id}/analysis`

Retrieves comprehensive analysis after interview completion.

**Response:**
```json
{
  "session_id": "uuid-here",
  "overall_score": 76.0,
  "total_questions": 5,
  "answered_questions": 5,
  "evaluations": [
    {
      "question_id": 0,
      "question_text": "Tell me about yourself and your background.",
      "user_answer": "User's answer text...",
      "strengths": ["Clear structure (STAR method)", "Relevant examples"],
      "improvements": ["Add more specific metrics", "Be more concise"],
      "score": 87.0,
      "feedback": "✓ Good use of examples • Consider adding more specific metrics • Show more impact"
    }
  ],
  "overall_feedback": "Great job! Your answers showed good structure and relevant examples. Focus on being more concise and adding more specific metrics to demonstrate impact.",
  "strengths_summary": ["Clear structure (STAR method)", "Relevant examples", "Good communication"],
  "improvement_areas": ["Add more specific metrics", "Be more concise", "Show more impact"],
  "recommendations": ["Practice answering questions out loud", "Prepare examples using STAR method"],
  "completed_at": "2024-01-01T00:00:00"
}
```

---

### 5. Get Session Status
**GET** `/interview/{session_id}/status`

Check the current status of an interview session.

**Response:**
```json
{
  "session_id": "uuid-here",
  "is_complete": false,
  "current_question": 2,
  "total_questions": 5,
  "progress_percentage": 40.0,
  "answers_submitted": 2,
  "created_at": "2024-01-01T00:00:00"
}
```

---

## Interview Flow

1. **User fills form** → Node.js server calls `POST /interview/create`
2. **Display question** → Node.js server calls `GET /interview/{session_id}/question`
3. **User submits answer** → Node.js server calls `POST /interview/{session_id}/answer`
4. **Repeat steps 2-3** until `is_complete: true`
5. **Show analysis** → Node.js server calls `GET /interview/{session_id}/analysis`

## Error Handling

All endpoints return standard HTTP status codes:
- `200` - Success
- `201` - Created (for session creation)
- `400` - Bad Request (invalid input, interview complete, etc.)
- `404` - Not Found (session doesn't exist)
- `500` - Internal Server Error

Error response format:
```json
{
  "detail": "Error message here"
}
```

