# app/vector_db/search_engine.py

from app.vector_db.user_history import UserHistoryManager
from app.embeddings.generator import EmbeddingGenerator
from app.vector_db.orm import VectorORM


class VectorSearchEngine:
    """Search predefined + user long-term memories and rank by relevance."""

    def __init__(self):
        # Keep predefined DB for static memory (optional)
        self.db = VectorORM()
        self.embed = EmbeddingGenerator()

        # Use the CORRECT LTM system
        self.history = UserHistoryManager()

    def search_relevant_chunks(self, query: str, user_id: str):
        print("🔍 [DEBUG] Searching LTM for user_id:", user_id)

        if not query.strip():
            return []

        # Create embedding once
        emb = self.embed.create_embedding(query)

        # --------------------------
        # 1) Predefined memory search
        # --------------------------
        try:
            predefined = self.db.search(self.db.predefined, emb, limit=5)
        except:
            predefined = []

        # --------------------------
        # 2) USER long-term memory search (Correct Path)
        # --------------------------
        try:
            user_mem = self.history.search_relevant_chunks(query, str(user_id))
        except:
            user_mem = []

        # user_mem already has text, score, source, metadata fields.

        # --------------------------
        # 3) Merge & sort both
        # --------------------------
        merged = (predefined or []) + (user_mem or [])

        for item in merged:
            item["final_score"] = float(item.get("score", 0.0))

        merged.sort(key=lambda x: x["final_score"], reverse=True)

        return merged
