# app/voice_interview/conversation_manager.py

from typing import List, Dict, Optional
from app.voice_interview.schemas import (
    VoiceInterviewConfigRequest,
    InterviewRole
)
from app.text_interview.schemas import InterviewQuestion
from app.llm.gemini_client import GeminiClient


class VoiceConversationManager:
    """
    Manages the conversation flow for voice interviews.
    Handles greetings, questions, follow-ups, and wrap-up.
    """

    def __init__(self):
        self.llm_client = GeminiClient()

    def generate_greeting(
        self,
        config: VoiceInterviewConfigRequest,
        interviewer_role: InterviewRole
    ) -> str:
        """
        Generate a personalized greeting for the interview.
        
        Args:
            config: Interview configuration
            interviewer_role: Role of the AI interviewer
            
        Returns:
            Greeting text
        """
        role_persona = self._get_role_persona(interviewer_role)
        
        prompt = f"""You are a {role_persona} conducting a {config.interview_type.value} interview.

Generate a warm, professional greeting (2-3 sentences) that:
1. Welcomes the candidate
2. Introduces yourself briefly
3. Sets expectations for the interview
4. Mentions the role: {config.job_role}

Keep it natural and conversational, as if speaking to the candidate in person.
Return ONLY the greeting text, no markdown, no quotes."""

        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=256)
            text = self.llm_client.extract_text(response)
            return text.strip().strip('"').strip("'") or self._get_default_greeting(config, interviewer_role)
        except Exception as e:
            print(f"⚠️ Error generating greeting: {e}")
            return self._get_default_greeting(config, interviewer_role)

    def generate_next_question(
        self,
        config: VoiceInterviewConfigRequest,
        question: InterviewQuestion,
        conversation_history: List[Dict],
        is_first_question: bool = False
    ) -> str:
        """
        Generate the next question with natural phrasing.
        
        Args:
            config: Interview configuration
            question: The question to ask
            conversation_history: Previous conversation turns (list of dicts with 'speaker' and 'text')
            is_first_question: Whether this is the first question
            
        Returns:
            Natural question text
        """
        role_persona = self._get_role_persona(config.interview_role)
        
        history_context = ""
        if conversation_history:
            recent_turns = conversation_history[-4:]
            history_context = "\n".join([
                f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}" for turn in recent_turns
            ])
        
        transition = "Let's begin with our first question." if is_first_question else "Let's move on to the next question."
        
        prompt = f"""You are a {role_persona} conducting a {config.interview_type.value} interview.

{transition}

Question to ask: {question.question_text}
Difficulty: {question.difficulty.value}

{history_context if history_context else "This is the start of the interview."}

Rephrase this question naturally as you would ask it in a real interview conversation.
Keep it conversational and friendly, but professional.
Return ONLY the rephrased question (1-2 sentences), no markdown, no quotes."""

        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=256)
            text = self.llm_client.extract_text(response)
            return text.strip().strip('"').strip("'") or question.question_text
        except Exception as e:
            print(f"⚠️ Error generating question phrasing: {e}")
            return question.question_text

    def generate_followup(
        self,
        config: VoiceInterviewConfigRequest,
        question: InterviewQuestion,
        user_answer: str,
        conversation_history: List[Dict]
    ) -> Optional[str]:
        """
        Generate a follow-up question or comment based on user's answer.
        
        Args:
            config: Interview configuration
            question: The original question
            user_answer: User's answer
            conversation_history: Previous conversation
            
        Returns:
            Follow-up text or None if no follow-up needed
        """
        answer_length = len(user_answer.split())
        
        if answer_length < 20:
            return "Could you tell me a bit more about that?"
        
        if answer_length > 300:
            return None
        
        role_persona = self._get_role_persona(config.interview_role)
        
        prompt = f"""You are a {role_persona} conducting a {config.interview_type.value} interview.

Original Question: {question.question_text}
Candidate's Answer: {user_answer}

Based on the candidate's answer, generate a brief follow-up question or comment (1 sentence) that:
1. Shows you're listening and engaged
2. Digs deeper into a specific point they mentioned
3. OR asks for clarification if needed
4. OR moves the conversation forward naturally

If the answer was comprehensive and complete, return "NONE" to indicate no follow-up is needed.
Otherwise, return ONLY the follow-up text (1 sentence), no markdown, no quotes."""

        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=128)
            text = self.llm_client.extract_text(response).strip().strip('"').strip("'")
            
            if text.upper() == "NONE" or len(text) < 10:
                return None
            
            return text
        except Exception as e:
            print(f"⚠️ Error generating follow-up: {e}")
            return None

    def generate_response(
        self,
        config: VoiceInterviewConfigRequest,
        user_message: str,
        conversation_history: List[Dict],
        current_phase: str,
        current_question: Optional[InterviewQuestion] = None
    ) -> str:
        """
        Generate AI response based on user input and conversation context.
        
        Args:
            config: Interview configuration
            user_message: User's transcribed speech
            conversation_history: Previous conversation turns
            current_phase: Current interview phase
            current_question: Current question being asked
            
        Returns:
            AI response text
        """
        role_persona = self._get_role_persona(config.interview_role)
        
        history_text = "\n".join([
            f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}" for turn in conversation_history[-6:]
        ])
        
        phase_guidance = self._get_phase_guidance(current_phase, current_question)
        
        prompt = f"""You are a {role_persona} conducting a {config.interview_type.value} interview for a {config.job_role} position.

Current Phase: {current_phase}
{phase_guidance}

Recent Conversation:
{history_text}

Candidate just said: {user_message}

Generate a natural, conversational response (1-3 sentences) that:
1. Acknowledges what they said appropriately
2. Moves the conversation forward
3. Maintains a professional but friendly tone
4. If they answered a question, provide brief positive feedback and either ask a follow-up or move to next question
5. If they asked a question, answer it briefly
6. If they seem confused, clarify

Return ONLY the response text, no markdown, no quotes."""

        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=256)
            text = self.llm_client.extract_text(response)
            return text.strip().strip('"').strip("'") or "I understand. Let's continue."
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return "Thank you for that. Let's continue."

    def generate_wrapup(
        self,
        config: VoiceInterviewConfigRequest,
        conversation_history: List[Dict]
    ) -> str:
        """
        Generate wrap-up message to conclude the interview.
        
        Args:
            config: Interview configuration
            conversation_history: Full conversation history
            
        Returns:
            Wrap-up text
        """
        role_persona = self._get_role_persona(config.interview_role)
        
        prompt = f"""You are a {role_persona} concluding a {config.interview_type.value} interview.

Generate a professional wrap-up message (2-3 sentences) that:
1. Thanks the candidate for their time
2. Provides next steps (e.g., "We'll review your responses and get back to you")
3. Ends on a positive, professional note

Keep it brief and professional.
Return ONLY the wrap-up text, no markdown, no quotes."""

        try:
            response = self.llm_client.generate_raw(prompt, max_output_tokens=256)
            text = self.llm_client.extract_text(response)
            return text.strip().strip('"').strip("'") or self._get_default_wrapup()
        except Exception as e:
            print(f"⚠️ Error generating wrap-up: {e}")
            return self._get_default_wrapup()

    def _get_role_persona(self, role: InterviewRole) -> str:
        """Get persona description for interviewer role."""
        personas = {
            InterviewRole.HR_RECRUITER: "friendly HR recruiter focused on cultural fit and soft skills",
            InterviewRole.TECHNICAL_INTERVIEWER: "technical interviewer focused on problem-solving and technical skills",
            InterviewRole.BEHAVIORAL_INTERVIEWER: "behavioral interviewer focused on past experiences and soft skills",
            InterviewRole.MIXED_INTERVIEWER: "professional interviewer covering both technical and behavioral aspects",
        }
        return personas.get(role, "professional interviewer")

    def _get_phase_guidance(self, phase: str, question: Optional[InterviewQuestion]) -> str:
        """Get guidance text for current interview phase."""
        if phase == "greeting":
            return "You are welcoming the candidate and setting up the interview."
        elif phase == "questions":
            if question:
                return f"You are asking questions. Current question: {question.question_text}"
            return "You are in the question phase of the interview."
        elif phase == "followup":
            return "You are asking follow-up questions or providing feedback."
        elif phase == "wrapup":
            return "You are concluding the interview."
        else:
            return "You are in a natural conversation flow."

    def _get_default_greeting(self, config: VoiceInterviewConfigRequest, role: InterviewRole) -> str:
        """Default greeting if LLM fails."""
        return f"Hello! Thank you for joining us today. I'm excited to learn more about your background and discuss the {config.job_role} position. Let's get started."

    def _get_default_wrapup(self) -> str:
        """Default wrap-up if LLM fails."""
        return "Thank you for taking the time to speak with us today. We really enjoyed learning more about you. We'll review your responses and be in touch soon. Have a great day!"

