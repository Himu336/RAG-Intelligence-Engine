# vector_db/search_engine.py

from vector_db.orm import VectorORM
from embeddings.generator import EmbeddingGenerator


class VectorSearchEngine:
    """
    Combines predefined context + long-term memory (user_history)
    and returns the most relevant chunks for the RAG prompt.
    """

    def __init__(self):
        self.db = VectorORM()
        self.embed = EmbeddingGenerator()

    def search_relevant_chunks(self, query: str, user_id: str):
        if not query or len(query.strip()) < 2:
            return []

        # Create embedding for query
        emb = self.embed.create_embedding(query)

        # 1) Predefined context search (global knowledge about coach behavior)
        try:
            predefined = self.db.search(
                collection=self.db.predefined,
                embedding=emb,
                limit=5
            )
        except Exception:
            predefined = []

        # 2) User-specific memory
        try:
            user_mem = self.db.search(
                collection=self.db.user_history,
                embedding=emb,
                limit=5,
                user_id=user_id
            )
        except Exception:
            user_mem = []

        # Merge lists
        merged = (predefined or []) + (user_mem or [])

        # Add final_score for consistent sorting
        for m in merged:
            m["final_score"] = float(m.get("score", 0.0))

        # Sort descending relevance
        merged.sort(key=lambda x: x["final_score"], reverse=True)

        return merged
