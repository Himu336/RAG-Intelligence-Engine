import google.generativeai as genai
from app.config import settings


# Configure Gemini globally (used only for embeddings)
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY not set – embedding generation will be disabled.")


class EmbeddingGenerator:
    """
    Wrapper for Google's embedding model.
    Produces vector embeddings for search + memory systems.
    """

    def __init__(self, model: str = "models/text-embedding-004"):
        self.model = model

    def create_embedding(self, text: str):
        """
        Generate an embedding vector from text using Gemini.
        Returns [] if invalid text or API error.
        """
        if not text or not text.strip():
            return []

        if not settings.GEMINI_API_KEY:
            print("[Embedding WARNING] GEMINI_API_KEY not configured; returning empty embedding.")
            return []

        try:
            resp = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
        except Exception as e:
            print(f"[Embedding ERROR] {e}")
            return []

        # Gemini returns {"embedding": [...vector...]}
        return resp.get("embedding", [])
