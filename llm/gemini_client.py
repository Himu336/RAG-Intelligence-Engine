# llm/gemini_client.py
import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiClient:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model

    def generate(self, prompt: str):
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text
