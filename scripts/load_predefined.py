import json
import uuid
from vector_db.orm import VectorORM
from embeddings.generator import EmbeddingGenerator

DATA_FILE = "scripts/predefined_data.json"

def recreate_collection(db: VectorORM, name: str):
    print(f"🗑 Deleting old collection: {name}")
    try:
        if db.client.collection_exists(name):
            db.client.delete_collection(name)
    except:
        pass

    print(f"📌 Recreating collection: {name}")
    db._ensure_collection(name)


def load_from_list(db, emb, items):
    print(f"📥 Preparing to insert {len(items)} predefined items...\n")

    for item in items:
        text = item["text"]
        role = item.get("role", "system")

        embedding = emb.create_embedding(text)

        db.insert(
            collection=db.predefined,
            text=text,
            embedding=embedding,
            metadata={
                "role": role
            }
        )

    print("✅ Predefined context loaded!")


def load_from_json(path):
    print("🔄 Loading predefined context into Qdrant...")
    
    db = VectorORM()
    emb = EmbeddingGenerator()

    recreate_collection(db, db.predefined)

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    load_from_list(db, emb, items)


if __name__ == "__main__":
    load_from_json(DATA_FILE)
