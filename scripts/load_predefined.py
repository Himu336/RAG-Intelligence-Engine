from vector_db.orm import VectorORM
from embeddings.generator import EmbeddingGenerator
import uuid

data = context_chunks = [

]

embedder = EmbeddingGenerator()
db = VectorORM()

for text in data:
    emb = embedder.create_embedding(text)
    db.insert(
        collection=db.predefined,
        text=text,
        embedding=emb,
        metadata={"id": str(uuid.uuid4())}
    )

print("Predefined context loaded.")
