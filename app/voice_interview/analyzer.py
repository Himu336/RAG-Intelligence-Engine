# app/voice_interview/analyzer.py

import json
import re
from datetime import datetime
from typing import List, Dict
from json import JSONDecodeError

# optional JSON repair library
try:
    from json_repair import repair_json
except:
    repair_json = None

from app.voice_interview.schemas import VoiceInterviewAnalysis
from app.llm.gemini_client import GeminiClient


def safe_json_parse(raw_text: str):
    """
    Bulletproof JSON parser:
    - Extracts JSON even if LLM adds extra text
    - Repairs broken JSON (quotes, commas, braces)
    - Never throws — always returns dict/list/None
    """

    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()

    # Remove markdown ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    # Extract JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    # Try direct parse
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    # Try repair if library available
    if repair_json:
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            pass

    # Remove trailing commas
    text = re.sub(r",\s*(\]|\})", r"\1", text)

    try:
        return json.loads(text)
    except Exception:
        return None


class VoiceInterviewAnalyzer:
    """
    Analyzes voice interview sessions and generates comprehensive evaluations.
    Evaluates communication skills, content quality, and overall performance.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    def generate_analysis(
        self,
        session_id: str,
        conversation_turns: List[Dict],
        config: dict
    ) -> VoiceInterviewAnalysis:

        if not conversation_turns:
            return self._create_empty_analysis(session_id, config)

        user_turns = [turn for turn in conversation_turns if turn.get("speaker") == "user"]
        ai_turns = [turn for turn in conversation_turns if turn.get("speaker") == "ai"]

        if not user_turns:
            return self._create_empty_analysis(session_id, config)

        conversation_text = self._build_conversation_text(conversation_turns)

        try:
            analysis_data = self._generate_llm_analysis(
                conversation_text,
                user_turns,
                ai_turns,
                config
            )

            return VoiceInterviewAnalysis(
                session_id=session_id,
                overall_score=analysis_data.get("overall_score", 70.0),
                total_questions=len(ai_turns),
                answered_questions=len(user_turns),
                conversation_summary=analysis_data.get("conversation_summary", ""),
                strengths_summary=analysis_data.get("strengths", []),
                improvement_areas=analysis_data.get("improvements", []),
                recommendations=analysis_data.get("recommendations", []),
                communication_score=analysis_data.get("communication_score", 70.0),
                content_score=analysis_data.get("content_score", 70.0),
                engagement_score=analysis_data.get("engagement_score", 70.0),
                completed_at=datetime.utcnow()
            )

        except Exception as e:
            print(f"❌ Error generating analysis: {e}")
            return self._create_fallback_analysis(session_id, conversation_turns, config)

    def _generate_llm_analysis(
        self,
        conversation_text: str,
        user_turns: List[Dict],
        ai_turns: List[Dict],
        config: dict
    ) -> dict:

        job_role = config.get("job_role", "the role") or "the role"
        experience_level = config.get("experience_level", {}).get("value", "") if isinstance(config.get("experience_level"), dict) else str(config.get("experience_level", ""))
        interview_type = config.get("interview_type", {}).get("value", "") if isinstance(config.get("interview_type"), dict) else str(config.get("interview_type", ""))

        if len(conversation_text) > 8000:
            conversation_text = conversation_text[:4000] + "\n... [truncated] ...\n" + conversation_text[-4000:]

        prompt = f"""You are evaluating a voice interview performance.

Job Role: {job_role}
Experience Level: {experience_level}
Interview Type: {interview_type}

=== CONVERSATION TRANSCRIPT ===
{conversation_text}

=== EVALUATION TASK ===
Analyze the candidate's performance across three dimensions:

1. **Communication Score (0-100)**: Clarity, articulation, pace, tone, professional language
2. **Content Score (0-100)**: Relevance, depth, examples, technical accuracy, completeness
3. **Engagement Score (0-100)**: Enthusiasm, confidence, listening, responsiveness, presence

Provide:
- Overall score (average of the three)
- Conversation summary (3-4 sentences)
- Top 3-5 strengths
- Top 3-5 improvement areas
- 3-5 actionable recommendations

Return STRICT JSON format:
{{
  "overall_score": 75.5,
  "communication_score": 80.0,
  "content_score": 72.0,
  "engagement_score": 74.5,
  "conversation_summary": "Summary...",
  "strengths": ["a", "b"],
  "improvements": ["c", "d"],
  "recommendations": ["e", "f"]
}}

Return ONLY valid JSON.
"""

        response = self.llm_client.generate_raw(prompt, max_output_tokens=2048)
        text = self.llm_client.extract_text(response)

        analysis = safe_json_parse(text)

        if not analysis:
            raise Exception("LLM returned invalid JSON")

        analysis["overall_score"] = self._normalize_score(analysis.get("overall_score", 70.0))
        analysis["communication_score"] = self._normalize_score(analysis.get("communication_score", 70.0))
        analysis["content_score"] = self._normalize_score(analysis.get("content_score", 70.0))
        analysis["engagement_score"] = self._normalize_score(analysis.get("engagement_score", 70.0))

        analysis["strengths"] = analysis.get("strengths", [])[:5]
        analysis["improvements"] = analysis.get("improvements", [])[:5]
        analysis["recommendations"] = analysis.get("recommendations", [])[:5]

        return analysis

    def _build_conversation_text(self, turns: List[Dict]) -> str:
        lines = []
        for turn in turns:
            speaker_label = "Interviewer" if turn.get("speaker") == "ai" else "Candidate"
            lines.append(f"{speaker_label}: {turn.get('text', '')}")
        return "\n".join(lines)

    def _clean_json(self, text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            if lines[-1].strip().startswith("```"):
                t = "\n".join(lines[1:-1])
            else:
                t = "\n".join(lines[1:])
            t = t.strip()

        start = t.find("{")
        end = t.rfind("}") + 1
        if start >= 0 and end > start:
            return t[start:end]

        return t

    def _normalize_score(self, score: any) -> float:
        try:
            score = float(score)
            return max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            return 70.0

    def _create_empty_analysis(
        self,
        session_id: str,
        config: dict
    ) -> VoiceInterviewAnalysis:

        return VoiceInterviewAnalysis(
            session_id=session_id,
            overall_score=0.0,
            total_questions=0,
            answered_questions=0,
            conversation_summary="No conversation recorded.",
            strengths_summary=[],
            improvement_areas=["Complete the interview to receive feedback"],
            recommendations=[
                "Practice speaking clearly and confidently",
                "Prepare examples using the STAR method",
                "Research the company and role thoroughly"
            ],
            communication_score=0.0,
            content_score=0.0,
            engagement_score=0.0,
            completed_at=datetime.utcnow()
        )

    def _create_fallback_analysis(
        self,
        session_id: str,
        conversation_turns: List[Dict],
        config: dict
    ) -> VoiceInterviewAnalysis:

        user_turns = [turn for turn in conversation_turns if turn.get("speaker") == "user"]
        ai_turns = [turn for turn in conversation_turns if turn.get("speaker") == "ai"]

        total_words = sum(len(turn.get("text", "").split()) for turn in user_turns)
        avg_words_per_turn = total_words / len(user_turns) if user_turns else 0

        communication_score = min(85, max(50, 60 + (avg_words_per_turn / 5)))
        content_score = min(85, max(50, 60 + (total_words / 50)))
        engagement_score = min(85, max(50, 70))
        overall_score = (communication_score + content_score + engagement_score) / 3

        strengths = []
        improvements = []

        if avg_words_per_turn > 50:
            strengths.append("Provided detailed responses")
        else:
            improvements.append("Consider providing more detailed answers")

        if len(user_turns) >= len(ai_turns) * 0.8:
            strengths.append("Engaged well with the interviewer")
        else:
            improvements.append("Try to engage more actively")

        return VoiceInterviewAnalysis(
            session_id=session_id,
            overall_score=round(overall_score, 2),
            total_questions=len(ai_turns),
            answered_questions=len(user_turns),
            conversation_summary=f"Interview completed with {len(user_turns)} responses to {len(ai_turns)} questions.",
            strengths_summary=strengths if strengths else ["Completed the interview"],
            improvement_areas=improvements if improvements else ["Continue practicing"],
            recommendations=[
                "Practice speaking clearly and at a comfortable pace",
                "Prepare specific examples from your experience",
                "Listen carefully to questions before responding"
            ],
            communication_score=round(communication_score, 2),
            content_score=round(content_score, 2),
            engagement_score=round(engagement_score, 2),
            completed_at=datetime.utcnow()
        )
