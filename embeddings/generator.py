import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

class EmbeddingGenerator:
    def __init__(self, model: str = "models/text-embedding-004"):
        self.model = model

    def create_embedding(self, text: str):
        if not text or text.strip() == "":
            return []

        resp = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )

        # THIS IS THE FIX 🔥
        return resp["embedding"]
