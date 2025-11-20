# app/embeddings/generator.py

import google.generativeai as genai
from app.config import settings


# Configure Gemini globally
genai.configure(api_key=settings.GEMINI_API_KEY)


class EmbeddingGenerator:
    """
    Wrapper for Google's embedding model.
    Produces vector embeddings for search + memory systems.
    """

    def __init__(self, model: str = "models/text-embedding-004"):
        self.model = model

    def create_embedding(self, text: str):
        """
        Generate an embedding vector from text.
        Returns [] if invalid text or API error.
        """
        if not text or not text.strip():
            return []

        try:
            resp = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
        except Exception as e:
            print(f"[Embedding ERROR] {e}")
            return []

        # FIX: Gemini returns {"embedding": [...vector...]}
        return resp.get("embedding", [])
