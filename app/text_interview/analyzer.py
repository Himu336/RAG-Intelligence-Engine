# app/text_interview/analyzer.py

import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from app.text_interview.schemas import (
    InterviewAnalysis,
    AnswerEvaluation,
    InterviewQuestion
)
from app.llm.gemini_client import GeminiClient


class InterviewAnalyzer:
    """
    Professional interview analysis generator.
    Evaluates answers and provides comprehensive feedback.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    @staticmethod
    def _safe_get_config_value(config: dict, key: str, default: str = "") -> str:
        """Safely extract config value, handling both string and dict formats."""
        value = config.get(key, default)
        if isinstance(value, dict):
            return value.get("value", default)
        return str(value) if value else default

    @staticmethod
    def _format_feedback(strengths: List[str], improvements: List[str]) -> str:
        """Format feedback string with checkmarks and bullet points."""
        parts: List[str] = []
        clean_strengths = [s for s in strengths if s]
        clean_improvements = [i for i in improvements if i]

        if clean_strengths:
            parts.append(f"✓ {clean_strengths[0]}")
        if clean_improvements:
            for imp in clean_improvements[:2]:
                parts.append(f"• {imp}")
        return " ".join(parts) if parts else "✓ Answer provided • Consider adding more detail"

    @staticmethod
    def _truncate_answer(answer: str, max_words: int = 500) -> str:
        """Smart truncation: keep first N words + last 100 words for context."""
        words = answer.split()
        if len(words) <= max_words:
            return answer
        
        # Keep first 500 words + last 100 words
        first_part = " ".join(words[:max_words])
        last_part = " ".join(words[-100:])
        return f"{first_part}... [truncated] ...{last_part}"

    @staticmethod
    def _normalize_score(raw_score: Any, default: float = 50.0) -> float:
        """
        Normalize score to 0–100 range without guessing the scale too aggressively.
        - Try to parse as float
        - Clamp between 0 and 100
        """
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = default

        if score < 0:
            score = 0.0
        if score > 100:
            score = 100.0
        return score

    def generate_analysis(
        self,
        session_id: str,
        questions: List[InterviewQuestion],
        answers: Dict[int, str],
        config: dict
    ) -> InterviewAnalysis:
        """
        Generate comprehensive analysis of the interview session.
        OPTIMIZED: Uses batch evaluation to reduce API calls from 6 to 1.
        """
        # Filter questions that have answers
        qa_pairs: List[Tuple[InterviewQuestion, str]] = []
        for question in questions:
            answer_text = answers.get(question.question_id, "")
            if answer_text:
                truncated_answer = self._truncate_answer(answer_text)
                qa_pairs.append((question, truncated_answer))
        
        if not qa_pairs:
            # No answers provided
            return InterviewAnalysis(
                session_id=session_id,
                overall_score=0.0,
                total_questions=len(questions),
                answered_questions=0,
                evaluations=[],
                overall_feedback="No answers were submitted for evaluation.",
                strengths_summary=[],
                improvement_areas=["Complete the interview by answering all questions"],
                recommendations=["Practice answering interview questions", "Prepare examples using STAR method"],
                completed_at=datetime.utcnow()
            )
        
        print(f"\n🚀 SINGLE BATCH CALL: Evaluating {len(qa_pairs)} answers + overall analysis")
        
        try:
            result = self._evaluate_all_in_one_call(qa_pairs, config)
            evaluations = result["evaluations"]
            overall_analysis = result["overall_analysis"]
            print(f"✅ SINGLE BATCH SUCCESS - {len(evaluations)} evaluations + overall analysis")
        except Exception as e:
            print(f"\n❌ SINGLE BATCH FAILED: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print(f"⚠️ Falling back to individual evaluations")
            # Fallback: individual evaluations
            evaluations: List[AnswerEvaluation] = []
            for question, answer_text in qa_pairs:
                evaluation = self._evaluate_answer(question, answer_text, config)
                evaluations.append(evaluation)
            
            overall_analysis = self._generate_overall_analysis(
                questions,
                evaluations,
                config
            )
        
        # Calculate overall score (already out of 100)
        if evaluations:
            overall_score = sum(e.score for e in evaluations) / len(evaluations)
        else:
            overall_score = 0.0
        
        return InterviewAnalysis(
            session_id=session_id,
            overall_score=round(overall_score, 2),
            total_questions=len(questions),
            answered_questions=len(evaluations),
            evaluations=evaluations,
            overall_feedback=overall_analysis.get("feedback", ""),
            strengths_summary=overall_analysis.get("strengths", []),
            improvement_areas=overall_analysis.get("improvements", []),
            recommendations=overall_analysis.get("recommendations", []),
            completed_at=datetime.utcnow()
        )

    def _evaluate_all_in_one_call(
        self,
        qa_pairs: List[tuple],
        config: dict
    ) -> dict:
        """
        Evaluate all answers + overall analysis in ONE single call.
        """
        prompt = self._build_simple_batch_prompt(qa_pairs, config)
        
        try:
            response = self.llm_client.generate_raw(
                prompt,
                max_output_tokens=4000
            )
            text = self.llm_client.extract_text(response)
            
            if not text or len(text) < 50:
                if getattr(response, "candidates", None):
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason == 2:
                        safety_ratings = getattr(candidate, 'safety_ratings', [])
                        raise Exception(f"Response blocked by safety filters. Safety: {safety_ratings}")
                raise Exception(f"Empty response (length: {len(text) if text else 0})")
            
            print(f"📝 Response length: {len(text)} chars")
            
            result_data = self._parse_simple_batch_json(text, qa_pairs)
            
            evaluations: List[AnswerEvaluation] = []
            eval_list = result_data.get("evaluations", [])
            for idx, (question, answer) in enumerate(qa_pairs):
                eval_data = eval_list[idx] if idx < len(eval_list) else {}

                strengths = eval_data.get("strengths") or ["Answer provided"]
                improvements = eval_data.get("improvements") or ["Could be more detailed"]
                raw_score = eval_data.get("score", 50.0)
                score = self._normalize_score(raw_score)

                feedback = eval_data.get("feedback", "")
                if not feedback:
                    feedback = self._format_feedback(strengths, improvements)

                evaluations.append(AnswerEvaluation(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    user_answer=answer,
                    strengths=strengths,
                    improvements=improvements,
                    score=round(score, 0),
                    feedback=feedback
                ))
            
            return {
                "evaluations": evaluations,
                "overall_analysis": result_data.get("overall_analysis", {
                    "feedback": "Analysis completed.",
                    "strengths": [],
                    "improvements": [],
                    "recommendations": []
                })
            }
            
        except Exception as e:
            print(f"❌ Error in single batch call: {e}")
            raise

    def _build_simple_batch_prompt(
        self,
        qa_pairs: List[tuple],
        config: dict
    ) -> str:
        """Build simple, clear batch evaluation prompt."""
        
        job_role = config.get("job_role", "the role") or "the role"
        experience_level = self._safe_get_config_value(config, "experience_level")
        interview_type = self._safe_get_config_value(config, "interview_type")
        
        qa_sections: List[str] = []
        for idx, (question, answer) in enumerate(qa_pairs):
            qa_sections.append(f"""
Q{idx+1} (question_id: {question.question_id}, Difficulty: {question.difficulty.value}):
Question: {question.question_text}
Answer: {answer}
""")
        
        qa_text = "\n".join(qa_sections)
        
        prompt = f"""You are evaluating {len(qa_pairs)} interview answers and providing overall analysis.

Job: {job_role} | Level: {experience_level} | Type: {interview_type}

=== INTERVIEW ANSWERS ===
{qa_text}

=== TASK ===
1. Evaluate EACH answer: Provide strengths (2-3), improvements (2-3), score (0-100), feedback string.
2. Provide overall analysis: summary, top strengths, top improvements, recommendations.

Return STRICT JSON in this format:
{{
  "evaluations": [
    {{
      "question_id": 0,
      "strengths": ["strength1", "strength2"],
      "improvements": ["improvement1", "improvement2"],
      "score": 75,
      "feedback": "✓ strength1 • improvement1 • improvement2"
    }}
  ],
  "overall_analysis": {{
    "feedback": "Overall performance summary (3-4 sentences)",
    "strengths": ["top strength 1", "top strength 2"],
    "improvements": ["improvement 1", "improvement 2"],
    "recommendations": ["recommendation 1", "recommendation 2"]
  }}
}}

Return ONLY valid JSON."""
        
        return prompt.strip()

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown ``` fences if present."""
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            if lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            t = "\n".join(lines).strip()
        return t

    def _parse_simple_batch_json(
        self,
        text: str,
        qa_pairs: List[tuple]
    ) -> dict:
        """Parse simple batch JSON response."""
        t = self._strip_code_fences(text)

        # Try direct parse
        try:
            if t.startswith("{"):
                return json.loads(t)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error (batch direct): {e}")

        # Try extracting object boundaries
        try:
            start_idx = t.find("{")
            end_idx = t.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = t[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error (batch extracted): {e}")
            print(f"Text preview: {t[:300]}")

        # Fallback
        return {
            "evaluations": [{} for _ in qa_pairs],
            "overall_analysis": {
                "feedback": "Error parsing response. Please try again.",
                "strengths": [],
                "improvements": [],
                "recommendations": []
            }
        }

    def _evaluate_answer(
        self,
        question: InterviewQuestion,
        answer: str,
        config: dict
    ) -> AnswerEvaluation:
        """Evaluate a single answer."""
        
        prompt = self._build_evaluation_prompt(question, answer, config)
        
        try:
            response = self.llm_client.generate_raw(
                prompt,
                max_output_tokens=1536
            )
            text = self.llm_client.extract_text(response)
            
            if not text or len(text) < 10:
                if getattr(response, "candidates", None):
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason == 2:
                        print(f"⚠️ LLM response blocked for question {question.question_id} - using fallback")
                        return self._create_fallback_evaluation(question, answer)
                print(f"⚠️ Empty response for question {question.question_id} - using fallback")
                return self._create_fallback_evaluation(question, answer)
            
            eval_data = self._parse_evaluation_json(text, question, answer)
            
            strengths = eval_data.get("strengths", [])
            improvements = eval_data.get("improvements", [])
            raw_score = eval_data.get("score", 50.0)
            score = self._normalize_score(raw_score)

            feedback = eval_data.get("feedback", "")
            if not feedback:
                feedback = self._format_feedback(strengths, improvements)
            
            return AnswerEvaluation(
                question_id=question.question_id,
                question_text=question.question_text,
                user_answer=answer,
                strengths=strengths,
                improvements=improvements,
                score=round(score, 0),
                feedback=feedback
            )
            
        except Exception as e:
            print(f"❌ Error evaluating answer: {e}")
            return self._create_fallback_evaluation(question, answer)

    def _build_evaluation_prompt(
        self,
        question: InterviewQuestion,
        answer: str,
        config: dict
    ) -> str:
        """Build prompt for evaluating a single answer."""
        
        job_role = config.get("job_role", "the role") or "the role"
        experience_level = self._safe_get_config_value(config, "experience_level")
        interview_type = self._safe_get_config_value(config, "interview_type")
        
        prompt = f"""Evaluate this interview answer.

Role: {job_role} | Level: {experience_level} | Type: {interview_type}

Q: {question.question_text} (Difficulty: {question.difficulty.value})
A: {answer}

Provide STRICT JSON:
{{
  "strengths": ["point1", "point2"],
  "improvements": ["point1", "point2"],
  "score": 87,
  "feedback": "✓ [strength] • [improvement1] • [improvement2]"
}}

Score: 0-100 (consider relevance, depth, clarity, examples, metrics). Return ONLY JSON."""
        return prompt.strip()

    def _parse_evaluation_json(
        self,
        text: str,
        question: InterviewQuestion,
        answer: str
    ) -> dict:
        """Parse evaluation JSON from LLM response."""
        original_text = text
        t = self._strip_code_fences(text)
        t = t.strip()
        
        print(f"\n{'='*60}")
        print(f"🔧 JSON PARSING ATTEMPT (single):")
        print(f"{'='*60}")
        print(f"Text length: {len(t)}")
        print(f"Starts with '{{': {t.startswith('{')}")
        
        start_idx = -1
        end_idx = 0
        
        try:
            if t.startswith("{"):
                parsed = json.loads(t)
                print(f"✅ JSON PARSING SUCCESS (direct parse)")
                print(f"Parsed score: {parsed.get('score', 'MISSING')}")
                return parsed

            start_idx = t.find("{")
            end_idx = t.rfind("}") + 1
            print(f"JSON extraction - start: {start_idx}, end: {end_idx}")
            if start_idx >= 0 and end_idx > start_idx:
                json_str = t[start_idx:end_idx]
                print(f"Extracted JSON string length: {len(json_str)}")
                parsed = json.loads(json_str)
                print(f"✅ JSON PARSING SUCCESS (extracted)")
                print(f"Parsed score: {parsed.get('score', 'MISSING')}")
                return parsed
            else:
                print(f"❌ JSON PARSING FAILED: Could not find JSON boundaries")
        except json.JSONDecodeError as e:
            print(f"❌ JSON PARSING FAILED: {e}")
            print(f"Error details: {str(e)}")
            if start_idx >= 0 and end_idx > start_idx:
                print(f"Attempted to parse: {t[start_idx:end_idx][:200]}")
            else:
                print(f"Attempted to parse: {t[:200]}")
        
        print(f"⚠️ Using FALLBACK evaluation (score: 50.0)")
        print(f"{'='*60}\n")
        
        # Fallback
        return {
            "strengths": ["Answer provided"],
            "improvements": ["Could be more detailed"],
            "score": 50.0,
            "feedback": "✓ Answer provided • Consider adding more specific examples • Include quantifiable metrics"
        }

    def _generate_overall_analysis(
        self,
        questions: List[InterviewQuestion],
        evaluations: List[AnswerEvaluation],
        config: dict
    ) -> dict:
        """Generate overall interview analysis."""
        
        if not evaluations:
            return {
                "feedback": "No answers were submitted for evaluation.",
                "strengths": [],
                "improvements": ["Complete the interview by answering all questions"],
                "recommendations": ["Practice answering interview questions", "Prepare examples using STAR method"]
            }
        
        all_strengths: List[str] = []
        all_improvements: List[str] = []
        scores: List[float] = []
        
        for eval in evaluations:
            all_strengths.extend(eval.strengths)
            all_improvements.extend(eval.improvements)
            scores.append(eval.score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        job_role = config.get('job_role', 'the role') or 'the role'
        experience_level = self._safe_get_config_value(config, 'experience_level')
        interview_type = self._safe_get_config_value(config, 'interview_type')
        
        prompt = f"""Provide overall interview analysis.

Role: {job_role} | Level: {experience_level} | Type: {interview_type}
Score: {avg_score:.0f}/100 | Answered: {len(evaluations)}/{len(questions)}

Strengths: {', '.join(all_strengths[:5])}
Improvements: {', '.join(all_improvements[:5])}

Return STRICT JSON:
{{
  "feedback": "3-4 sentence summary",
  "strengths": ["top3-5"],
  "improvements": ["top3-5"],
  "recommendations": ["actionable3-5"]
}}

Return ONLY JSON."""
        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=1024)
            text = self.llm_client.extract_text(response)
            
            t = self._strip_code_fences(text).strip()
            
            if t.startswith("{"):
                return json.loads(t)
            start_idx = t.find("{")
            end_idx = t.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(t[start_idx:end_idx])
        except Exception as e:
            print(f"⚠️ Error generating overall analysis: {e}")
        
        # Fallback
        return {
            "feedback": "Great job! Your answers showed good structure and relevant examples. Focus on being more concise and adding more specific metrics to demonstrate impact.",
            "strengths": list(set(all_strengths[:5])) if all_strengths else ["Clear structure", "Relevant examples"],
            "improvements": list(set(all_improvements[:5])) if all_improvements else ["Add more specific metrics", "Be more concise"],
            "recommendations": [
                "Practice answering questions out loud",
                "Prepare examples using the STAR method",
                "Research the company and role thoroughly"
            ]
        }

    def _create_fallback_evaluation(
        self,
        question: InterviewQuestion,
        answer: str
    ) -> AnswerEvaluation:
        """Create a basic fallback evaluation."""
        word_count = len(answer.split())
        
        strengths: List[str] = []
        improvements: List[str] = []
        
        if word_count > 50:
            strengths.append("Provided a detailed answer")
        else:
            improvements.append("Consider providing more detail")
        
        if "example" in answer.lower() or "experience" in answer.lower():
            strengths.append("Included relevant examples")
        else:
            improvements.append("Add specific examples from your experience")
        
        base_score = min(70, max(40, word_count * 2))
        if "example" in answer.lower() or "experience" in answer.lower():
            base_score += 10
        score = min(85, base_score)
        
        feedback = self._format_feedback(strengths or ["Answer provided"], improvements or ["Consider expanding your answer"])
        
        return AnswerEvaluation(
            question_id=question.question_id,
            question_text=question.question_text,
            user_answer=answer,
            strengths=strengths if strengths else ["Answer provided"],
            improvements=improvements if improvements else ["Consider expanding your answer"],
            score=round(score, 0),
            feedback=feedback
        )
