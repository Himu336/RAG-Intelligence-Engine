from embeddings.generator import EmbeddingGenerator
from vector_db.orm import VectorORM

class VectorSearchEngine:
    def __init__(self):
        self.db = VectorORM()
        self.embedder = EmbeddingGenerator()

    def search_user_history(self, user_id: str, embedding, top_k=4):
        col = self.db.client.get_collection(self.db.user_history)
        
        results = col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"user_id": user_id}  # Filter by user
        )

        return results

    def search_relevant_chunks(self, query: str, user_id: str, top_k=4):
        # Embed the query
        embedding = self.embedder.create_embedding(query)

        # Predefined content search
        predefined = self.db.search(self.db.predefined, embedding, top_k)

        # User-specific search
        user_history = self.search_user_history(user_id, embedding, top_k)

        merged = []

        # Format predefined
        if "documents" in predefined and predefined["documents"]:
            for idx in range(len(predefined["documents"][0])):
                merged.append({
                    "text": predefined["documents"][0][idx],
                    "score": predefined["distances"][0][idx]
                })

        # Format user history
        if "documents" in user_history and user_history["documents"]:
            for idx in range(len(user_history["documents"][0])):
                merged.append({
                    "text": user_history["documents"][0][idx],
                    "score": user_history["distances"][0][idx]
                })

        # Lower cosine score = better match
        merged_sorted = sorted(merged, key=lambda x: x["score"])

        return merged_sorted[:top_k]
