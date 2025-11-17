# vector_db/orm.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid
from app.config import settings

EMBEDDING_DIM = 768


class VectorORM:
    def __init__(self):
        # Use Qdrant Cloud URL + API key from settings.py / env
        self.client = QdrantClient(
            url=settings.QDRANT_URL,          # Example: https://xxxxxx.us-east-1.aws.cloud.qdrant.io
            api_key=settings.QDRANT_API_KEY,  # Your cloud API key
            prefer_grpc=False                 # IMPORTANT for cloud
        )

        self.predefined = settings.PREDEFINED_COLLECTION
        self.user_history = settings.USER_HISTORY_COLLECTION

        # Ensure collections exist
        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    def _ensure_collection(self, name):
        """Create collection on cloud if missing."""
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

    def insert(self, collection, text, embedding, metadata):
        """Insert 1 item into Qdrant."""
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": text,
                **metadata
            }
        )
        self.client.upsert(collection_name=collection, points=[point])

    def search(self, collection, embedding, limit=5, user_id=None):
        """Search with optional user filter."""
        payload_filter = None
        if user_id:
            payload_filter = Filter(
                must=[FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )]
            )

        results = self.client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit,
            query_filter=payload_filter
        )

        out = []
        for r in results:
            out.append({
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source": collection,
                "metadata": r.payload
            })
        return out

    def delete(self, collection, point_id):
        """Remove a point by ID."""
        self.client.delete(
            collection_name=collection,
            points_selector={"points": [point_id]}
        )

    # Custom query wrapper used by user_history
    def query(self, collection, query_vector, limit=5, where=None):
        """
        where = {"user_id": "123"} OR {"type": "summary"} etc.
        """
        payload_filter = None

        if where:
            must_conditions = []
            for key, val in where.items():
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=val)
                    )
                )

            payload_filter = Filter(must=must_conditions)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=payload_filter
        )

        formatted = []
        for r in results:
            formatted.append({
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source": collection,
                "metadata": r.payload
            })
        return formatted
