# app/text_interview/generator.py

import json
import re
from typing import List, Optional
from app.text_interview.schemas import (
    InterviewConfigRequest,
    InterviewQuestion,
    DifficultyLevel,
    QuestionHint,
    InterviewType
)
from app.llm.gemini_client import GeminiClient


class InterviewQuestionGenerator:
    """
    Professional interview question generator using Gemini LLM.
    Generates contextually relevant questions based on user specifications.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    def generate_questions(self, config: InterviewConfigRequest) -> List[InterviewQuestion]:
        """
        Generate interview questions based on configuration.
        
        Args:
            config: Interview configuration with job role, experience, etc.
            
        Returns:
            List of InterviewQuestion objects
        """
        prompt = self._build_generation_prompt(config)
        
        try:
            response = self.llm_client.generate_raw(
                prompt,
                max_output_tokens=2048
            )
            text = self.llm_client.extract_text(response)

            if hasattr(response, "candidates"):
                cand = response.candidates[0]
                
                if getattr(cand, "finish_reason", None) == 2:
                    print("⚠️ Gemini blocked output — retrying")
                    raise Exception("Gemini blocked output")

            questions_data = self._parse_questions_json(text, config)
            
            questions: List[InterviewQuestion] = []
            for idx, q_data in enumerate(questions_data):
                question = self._create_question_from_data(q_data, idx, config.interview_type)
                questions.append(question)
            
            return questions[:config.num_questions]
            
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            # Fallback: return default questions
            return self._generate_fallback_questions(config)

    def _build_generation_prompt(self, config: InterviewConfigRequest) -> str:
        """Build a detailed prompt for question generation."""
        
        job_desc_context = ""
        if config.job_description:
            # Limit to avoid token overflow
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

Generate {config.num_questions} high-quality, relevant interview questions for the following specifications:

Job Role: {config.job_role}
Experience Level: {config.experience_level.value}
{company_context}{job_desc_context}

{experience_guidance}

{interview_type_guidance}

Requirements:
1. Questions must be SPECIFIC to the job role "{config.job_role}" and experience level "{config.experience_level.value}"
2. Questions should be realistic and commonly asked in real interviews
3. Vary difficulty levels appropriately (mix of Easy, Medium, Hard)
4. Include helpful hints/frameworks where appropriate (e.g., STAR method for behavioral)
5. Make questions progressive - start easier, build complexity
6. If job description provided, reference specific requirements/skills mentioned
7. DO NOT use single quotes inside JSON strings. Escape or replace them.
8. DO NOT break lines inside strings.


Return STRICT JSON array format:
[
  {{
    "question_text": "Full question text here",
    "difficulty": "Easy|Medium|Hard",
    "hint": {{
      "text": "Helpful hint or framework suggestion",
      "framework": "STAR|CAR|etc (optional)"
    }},
    "interview_type": "{config.interview_type.value}"
  }},
  ...
]

Generate exactly {config.num_questions} questions. Return ONLY valid JSON, no markdown, no explanations."""
        return prompt.strip()

    def _get_interview_type_guidance(self, interview_type: InterviewType) -> str:
        """Get guidance text for different interview types."""
        guidance_map = {
            InterviewType.BEHAVIORAL: """
Behavioral Interview Focus:
- Past experiences and how you handled situations
- Leadership, teamwork, conflict resolution
- Problem-solving and decision-making
- Use STAR (Situation, Task, Action, Result) framework hints
- Examples: "Tell me about a time when...", "Describe a situation where..."
""",
            InterviewType.TECHNICAL: """
Technical Interview Focus:
- Programming concepts, algorithms, data structures
- System architecture and design patterns
- Technology-specific knowledge
- Problem-solving and coding challenges
- Examples: "Explain how...", "Design a system that...", "What's the time complexity of..."
""",
            InterviewType.SYSTEM_DESIGN: """
System Design Interview Focus:
- Scalability, reliability, performance
- Architecture patterns and trade-offs
- Distributed systems concepts
- Real-world system design scenarios
- Examples: "Design a system to...", "How would you scale...", "What are the trade-offs..."
""",
            InterviewType.MIXED: """
Mixed Interview Focus:
- Combination of behavioral, technical, and system design
- Balance between soft skills and technical knowledge
- Real-world scenarios requiring both
"""
        }
        return guidance_map.get(interview_type, "")

    def _get_experience_guidance(self, experience_level) -> str:
        """Get guidance based on experience level."""
        guidance_map = {
            "Entry Level (0-2 years)": """
Entry Level Expectations:
- Focus on learning ability, foundational knowledge
- Basic technical concepts and problem-solving
- Entry-level behavioral scenarios
- Questions should be approachable but still challenging
""",
            "Mid Level (3-5 years)": """
Mid Level Expectations:
- Practical experience and application
- Intermediate technical depth
- Leadership and collaboration examples
- Questions should test both knowledge and experience
""",
            "Senior Level (6-10 years)": """
Senior Level Expectations:
- Deep technical expertise and architecture
- Leadership, mentoring, strategic thinking
- Complex problem-solving and system design
- Questions should reflect senior responsibilities
""",
            "Executive Level (10+ years)": """
Executive Level Expectations:
- Strategic vision and business impact
- Leadership at scale, organizational impact
- High-level system design and architecture
- Questions should reflect executive-level thinking
"""
        }
        return guidance_map.get(experience_level.value, "")

    def _clean_json_str(self, text: str) -> str:
        """
        Cleans JSON to avoid parse failures caused by:
        - markdown fences
        - stray backticks
        - broken single quotes
        - unterminated strings
        - extra trailing commas
        - blocked/truncated Gemini outputs
        """
        if not text:
            return "[]"

        t = text.strip()

        # Remove markdown ```
        if t.startswith("```"):
            lines = t.split("\n")
            if lines[-1].strip().startswith("```"):
                t = "\n".join(lines[1:-1])
            else:
                t = "\n".join(lines[1:])
            t = t.strip()

        # Remove backticks
        t = t.replace("`", "")

        # Fix broken strings split across lines
        t = re.sub(r'"([^"\n]*)\n', r'"\n', t)

        # Extract only array part
        start = t.find("[")
        end = t.rfind("]") + 1
        if start != -1 and end > start:
            t = t[start:end]

        # Remove trailing commas before ] or }
        t = re.sub(r",\s*(\]|\})", r"\1", t)

        return t

    def _parse_questions_json(self, text: str, config: InterviewConfigRequest) -> List[dict]:
        """Parse JSON from LLM response with fallback handling."""
        if not text:
            return self._get_default_questions()

        cleaned = self._clean_json_str(text)

        # Try JSON parse after cleaning
        try:
            if cleaned.startswith("["):
                data = json.loads(cleaned)
                if isinstance(data, list) and data:
                    return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error (questions): {e}")
            print(f"Raw cleaned text preview: {cleaned[:500]}")

        # Fallback: try to parse as structured text
        return self._parse_fallback_format(text, default_type=config.interview_type.value)

    def _parse_fallback_format(self, text: str, default_type: str = "Mixed") -> List[dict]:
        """Fallback parser for non-JSON responses."""
        questions: List[dict] = []
        lines = text.split("\n")
        current_question = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Try to detect question patterns
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
                    "text": "Use the present-past-future framework. Start with current role, then relevant background, and future goals.",
                    "framework": "Present-Past-Future"
                },
                "interview_type": "Behavioral"
            },
            {
                "question_text": "Why are you interested in this role?",
                "difficulty": "Easy",
                "hint": {
                    "text": "Connect your skills and interests to the role requirements.",
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
        """Create InterviewQuestion from parsed data."""
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
        """Generate basic fallback questions if LLM fails."""
        questions: List[InterviewQuestion] = []
        base_questions = [
            {
                "question_text": f"Tell me about yourself and why you're interested in the {config.job_role} role.",
                "difficulty": DifficultyLevel.EASY,
                "hint": QuestionHint(
                    text="Use the present-past-future framework. Start with current role, then relevant background, and future goals.",
                    framework="Present-Past-Future"
                )
            },
            {
                "question_text": f"What experience do you have that makes you a good fit for {config.job_role}?",
                "difficulty": DifficultyLevel.MEDIUM,
                "hint": QuestionHint(
                    text="Provide specific examples from your experience that align with the role requirements.",
                    framework="STAR"
                )
            },
            {
                "question_text": f"Describe a challenging project or problem you've worked on and how you solved it.",
                "difficulty": DifficultyLevel.MEDIUM,
                "hint": QuestionHint(
                    text="Use the STAR method: Situation, Task, Action, Result.",
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
