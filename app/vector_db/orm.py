import uuid
from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

EMBEDDING_DIM = 768


class VectorORM:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False
        )

        self.predefined = settings.PREDEFINED_COLLECTION
        self.user_history = settings.USER_HISTORY_COLLECTION

        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    # ---------------------------------------------------------
    # SIMPLE COLLECTION CREATION — NO INDEXES, NO SCHEMA
    # ---------------------------------------------------------
    def _ensure_collection(self, name: str):
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            )

    # ---------------------------------------------------------
    # INSERT VECTOR
    # ---------------------------------------------------------
    def insert(self, collection, text, embedding, metadata):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text, **metadata}
        )
        self.client.upsert(collection, [point])

    # ---------------------------------------------------------
    # SEARCH WITH OPTIONAL FILTER
    # ---------------------------------------------------------
    def search(self, collection, embedding, limit=5, user_id=None):
        q_filter = None

        if user_id is not None:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=str(user_id))
                    )
                ]
            )

        results = self.client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit,
            query_filter=q_filter
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

    # ---------------------------------------------------------
    # DELETE VECTOR
    # ---------------------------------------------------------
    def delete(self, collection, point_id):
        self.client.delete(
            collection_name=collection,
            points_selector={"points": [point_id]}
        )

    # ---------------------------------------------------------
    # GENERIC QUERY (FILTER BY user_id / type)
    # ---------------------------------------------------------
    def query(self, collection, query_vector, limit=5, where=None):
        q_filter = None

        if where:
            conditions = [
                FieldCondition(
                    key=k,
                    match=MatchValue(value=str(v))
                ) for k, v in where.items()
            ]
            q_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=q_filter
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source": collection,
                "metadata": r.payload
            }
            for r in results
        ]
