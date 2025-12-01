# app/text_interview/generator.py

import json
import re
from typing import List, Optional
from json import JSONDecodeError

# optional json repair
try:
    from json_repair import repair_json
except:
    repair_json = None

from app.text_interview.schemas import (
    InterviewConfigRequest,
    InterviewQuestion,
    DifficultyLevel,
    QuestionHint,
    InterviewType
)
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

    # Remove markdown fences ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    # Extract array block
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        text = text[start:end]

    # Try direct JSON parse
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    # Try repair
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


class InterviewQuestionGenerator:
    """
    Professional interview question generator using Gemini LLM.
    Generates contextually relevant questions based on user specifications.
    """

    def __init__(self):
        self.llm_client = GeminiClient()
        self.JSON_PARSE_RETRY_LIMIT = 1

    def generate_questions(self, config: InterviewConfigRequest) -> List[InterviewQuestion]:
        """
        Generate interview questions based on configuration.
        """

        prompt = self._build_generation_prompt(config)
        last_text = ""

        for attempt in range(self.JSON_PARSE_RETRY_LIMIT + 1):
            try:
                response = self.llm_client.generate_raw(
                    prompt,
                    max_output_tokens=2048
                )
                text = self.llm_client.extract_text(response)
                last_text = text

                # Gemini block handling (log only)
                if hasattr(response, "candidates"):
                    cand = response.candidates[0]
                    if getattr(cand, "finish_reason", None) == 2:
                        print("⚠️ Gemini blocked output")

                parsed_payload = safe_json_parse(text)

                if isinstance(parsed_payload, list) and parsed_payload:
                    questions = self._build_questions_from_payload(
                        parsed_payload,
                        config.interview_type,
                        config.num_questions
                    )
                    if questions:
                        return questions

                if attempt < self.JSON_PARSE_RETRY_LIMIT:
                    print("⚠️ JSON parsing failed — retrying generation")
                    continue

                break

            except Exception as e:
                print(f"❌ Error generating questions: {e}")
                return self._generate_fallback_questions(config)

        print("⚠️ JSON parsing failed after retries. Using fallback parser.")
        fallback_payload = self._parse_fallback_format(
            last_text,
            default_type=config.interview_type.value
        )

        if fallback_payload:
            fallback_questions = self._build_questions_from_payload(
                fallback_payload,
                config.interview_type,
                config.num_questions
            )
            if fallback_questions:
                return fallback_questions

        return self._generate_fallback_questions(config)

    def _build_generation_prompt(self, config: InterviewConfigRequest) -> str:
        """Build a detailed prompt for question generation."""

        job_desc_context = ""
        if config.job_description:
            job_desc_limited = config.job_description[:1000]
            job_desc_context = f"""
Job Description:
{job_desc_limited}
"""

        company_context = ""
        if config.company:
            company_context = f"Target Company: {config.company}\n"

        interview_type_guidance = self._get_interview_type_guidance(config.interview_type)
        experience_guidance = self._get_experience_guidance(config.experience_level)

        prompt = f"""You are an expert interview coach specializing in {config.interview_type.lower()} interviews.

Generate {config.num_questions} high-quality, relevant interview questions for:

Job Role: {config.job_role}
Experience Level: {config.experience_level.value}
{company_context}{job_desc_context}

{experience_guidance}
{interview_type_guidance}

Return STRICT JSON array format:
[
  {{
    "question_text": "Full question text here",
    "difficulty": "Easy|Medium|Hard",
    "hint": {{
      "text": "Helpful hint or framework suggestion",
      "framework": "STAR|CAR|etc"
    }},
    "interview_type": "{config.interview_type.value}"
  }}
]

Return ONLY valid JSON, no markdown, no explanations.
"""

        return prompt.strip()

    def _get_interview_type_guidance(self, interview_type: InterviewType) -> str:
        """Get guidance text for different interview types."""
        guidance_map = {
            InterviewType.BEHAVIORAL: """
Behavioral Interview Focus:
- Past experiences and how you handled situations
- Leadership, teamwork, conflict resolution
- Use STAR (Situation, Task, Action, Result) framework hints
""",
            InterviewType.TECHNICAL: """
Technical Interview Focus:
- Programming concepts, algorithms, data structures
- System architecture and design patterns
- Technology-specific knowledge
""",
            InterviewType.SYSTEM_DESIGN: """
System Design Interview Focus:
- Scalability, reliability, performance
- Real-world system design scenarios
""",
            InterviewType.MIXED: """
Mixed Interview Focus:
- Combination of behavioral, technical, and system design
"""
        }
        return guidance_map.get(interview_type, "")

    def _get_experience_guidance(self, experience_level) -> str:
        """Get guidance based on experience level."""
        guidance_map = {
            "Entry Level (0-2 years)": """
Entry Level Expectations:
- Focus on learning ability and basic concepts
""",
            "Mid Level (3-5 years)": """
Mid Level Expectations:
- Practical experience and intermediate technical depth
""",
            "Senior Level (6-10 years)": """
Senior Level Expectations:
- Deep expertise and leadership
""",
            "Executive Level (10+ years)": """
Executive Level Expectations:
- Strategy, architecture, high-level decision making
"""
        }
        return guidance_map.get(experience_level.value, "")

    def _clean_json_str(self, text: str) -> str:
        """
        Legacy cleaning method (still used by fallback parser).
        """
        if not text:
            return "[]"

        t = text.strip()

        if t.startswith("```"):
            lines = t.split("\n")
            if lines[-1].strip().startswith("```"):
                t = "\n".join(lines[1:-1])
            else:
                t = "\n".join(lines[1:])
            t = t.strip()

        t = t.replace("`", "")
        t = re.sub(r'"([^"\n]*)\n', r'"\n', t)

        start = t.find("[")
        end = t.rfind("]") + 1
        if start != -1 and end > start:
            t = t[start:end]

        t = re.sub(r",\s*(\]|\})", r"\1", t)

        return t

    def _parse_fallback_format(self, text: str, default_type: str = "Mixed") -> List[dict]:
        """Fallback parser for non-JSON responses."""

        questions: List[dict] = []
        lines = text.split("\n")
        current_question = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "?" in line and len(line) > 20:
                if current_question:
                    questions.append(current_question)
                current_question = {
                    "question_text": line.rstrip("?").strip() + "?",
                    "difficulty": "Medium",
                    "hint": None,
                    "interview_type": default_type
                }

            elif current_question and ("hint" in line.lower() or "framework" in line.lower()):
                hint_text = (
                    line.replace("Hint:", "")
                    .replace("hint:", "")
                    .replace("Framework:", "")
                    .replace("framework:", "")
                    .strip()
                )
                if hint_text:
                    current_question["hint"] = {
                        "text": hint_text,
                        "framework": None
                    }

        if current_question:
            questions.append(current_question)

        return questions if questions else self._get_default_questions()

    def _get_default_questions(self) -> List[dict]:
        """Default fallback questions."""
        return [
            {
                "question_text": "Tell me about yourself and your background.",
                "difficulty": "Easy",
                "hint": {
                    "text": "Use present-past-future structure.",
                    "framework": "Present-Past-Future"
                },
                "interview_type": "Behavioral"
            },
            {
                "question_text": "Why are you interested in this role?",
                "difficulty": "Easy",
                "hint": {
                    "text": "Connect your skills to the job requirements.",
                    "framework": None
                },
                "interview_type": "Behavioral"
            }
        ]

    def _create_question_from_data(
        self,
        q_data: dict,
        idx: int,
        interview_type: InterviewType
    ) -> InterviewQuestion:

        difficulty_str = str(q_data.get("difficulty", "Medium")).title()
        try:
            difficulty = DifficultyLevel(difficulty_str)
        except ValueError:
            difficulty = DifficultyLevel.MEDIUM

        hint_data = q_data.get("hint")
        hint: Optional[QuestionHint] = None

        if hint_data:
            if isinstance(hint_data, dict):
                hint = QuestionHint(
                    text=hint_data.get("text", ""),
                    framework=hint_data.get("framework")
                )
            elif isinstance(hint_data, str):
                hint = QuestionHint(text=hint_data)

        return InterviewQuestion(
            question_id=idx,
            question_text=q_data.get("question_text", "Question not available"),
            difficulty=difficulty,
            hint=hint,
            interview_type=q_data.get("interview_type", interview_type.value)
        )

    def _generate_fallback_questions(self, config: InterviewConfigRequest) -> List[InterviewQuestion]:
        """Simple fallback if LLM fully fails."""
        questions: List[InterviewQuestion] = []
        base_questions = [
            {
                "question_text": f"Tell me about yourself and why you're interested in the {config.job_role} role.",
                "difficulty": DifficultyLevel.EASY,
                "hint": QuestionHint(
                    text="Use present-past-future structure.",
                    framework="Present-Past-Future"
                )
            },
            {
                "question_text": f"What experience do you have that makes you a good fit for {config.job_role}?",
                "difficulty": DifficultyLevel.MEDIUM,
                "hint": QuestionHint(
                    text="Use the STAR method.",
                    framework="STAR"
                )
            }
        ]

        for idx, q in enumerate(base_questions[:config.num_questions]):
            questions.append(InterviewQuestion(
                question_id=idx,
                question_text=q["question_text"],
                difficulty=q["difficulty"],
                hint=q["hint"],
                interview_type=config.interview_type.value
            ))

        return questions

    def _build_questions_from_payload(
        self,
        payload: List[dict],
        interview_type: InterviewType,
        limit: int
    ) -> List[InterviewQuestion]:
        """Convert parsed payload into InterviewQuestion objects."""
        questions: List[InterviewQuestion] = []
        for idx, q_data in enumerate(payload):
            question = self._create_question_from_data(q_data, idx, interview_type)
            questions.append(question)
            if len(questions) >= limit:
                break
        return questions
