from embeddings.generator import EmbeddingGenerator
from vector_db.orm import VectorORM
import uuid

class UserHistoryManager:
    def __init__(self):
        self.db = VectorORM()
        self.embedder = EmbeddingGenerator()

    def save_message(self, user_id: str, message: str):
        embedding = self.embedder.create_embedding(message)

        metadata = {
            "id": str(uuid.uuid4()),
            "user_id": user_id
        }

        self.db.insert(
            collection=self.db.user_history,
            text=message,
            embedding=embedding,
            metadata=metadata
        )

        return True
