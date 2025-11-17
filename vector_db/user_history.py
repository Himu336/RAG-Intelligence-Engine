# vector_db/user_history.py

import uuid
from typing import List, Dict
from vector_db.orm import VectorORM
from embeddings.generator import EmbeddingGenerator


def _is_trivial_text(s: str) -> bool:
    """Return True if text is too short or trivial to store as long-term memory."""
    if not s:
        return True
    s = s.strip().lower()
    # ignore very short greetings or confirmations
    trivial = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "sure", "no", "yes"}
    if s in trivial:
        return True
    if len(s) < 10:
        return True
    return False


class UserHistoryManager:
    """
    Long-term memory manager for your VectorORM-backed DB.
    - This version avoids saving raw greetings and small messages.
    - Save summaries only via save_summary() or upsert_summary().
    """

    def __init__(self):
        self.db = VectorORM()
        self.emb = EmbeddingGenerator()
        self.max_summaries = 6

    # Keep the compatibility method, but be conservative: only save non-trivial messages if explicitly desired
    def save_message(self, user_id: str, message: str, force: bool = False):
        """
        Save a raw message ONLY if it's not trivial or if force=True.
        By default we avoid storing short greetings or trivial lines.
        """
        if not force and _is_trivial_text(message):
            # skip saving trivial messages
            return

        embedding = self.emb.create_embedding(message)
        self.db.insert(
            collection=self.db.user_history,
            text=message,
            embedding=embedding,
            metadata={
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "type": "history"
            }
        )

    # Save distilled summary (always allowed)
    def save_summary(self, user_id: str, summary_text: str):
        if not summary_text or _is_trivial_text(summary_text):
            return
        embedding = self.emb.create_embedding(summary_text)
        self.db.insert(
            collection=self.db.user_history,
            text=summary_text,
            embedding=embedding,
            metadata={
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "type": "summary"
            }
        )

    def get_summaries(self, user_id: str) -> List[Dict]:
        # Uses underlying client scroll pattern (compatible with your ORM)
        try:
            results = self.db.client.scroll(
                collection_name=self.db.user_history,
                scroll_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}},
                        {"key": "type", "match": {"value": "summary"}}
                    ]
                },
                limit=100
            )
            points = results[0]
        except Exception:
            points = []

        out = []
        for p in points:
            out.append({
                "id": p.id,
                "text": p.payload.get("text", ""),
                "metadata": p.payload
            })
        return out

    def upsert_summary(self, user_id: str, summary_text: str):
        """
        Insert a new summary after checking for duplicates.
        Keeps only self.max_summaries most recent summaries.
        Deduplicate by exact text or strong substring match.
        """
        if not summary_text:
            return

        # Simple dedupe: if identical text already exists, skip
        existing = self.get_summaries(user_id)
        normalized_new = " ".join(summary_text.lower().split())
        for e in existing:
            if normalized_new == " ".join(e["text"].lower().split()):
                return  # duplicate exact

        # Also skip if new summary is substring of an existing summary
        for e in existing:
            if normalized_new in e["text"].lower():
                return

        # Save new summary
        self.save_summary(user_id, summary_text)

        # Trim old summaries to keep the collection small
        summaries = self.get_summaries(user_id)
        if len(summaries) > self.max_summaries:
            # delete the oldest (assume returned order is newest->oldest; adjust if necessary)
            # We'll delete the extra ones except the newest max_summaries
            to_delete = [s["id"] for s in summaries[self.max_summaries:]]
            for sid in to_delete:
                try:
                    self.db.delete(self.db.user_history, sid)
                except Exception:
                    pass

    def search_relevant_chunks(self, query: str, user_id: str, limit: int = 5):
        """Return relevant long-term memory (summaries & history), limit small."""
        embedding = self.emb.create_embedding(query)
        # Your VectorORM likely has a query method; keep the same contract
        results = self.db.query(
            collection=self.db.user_history,
            query_vector=embedding,
            limit=limit,
            where={"user_id": user_id}
        )
        chunks = []
        for match in results:
            chunks.append({
                "text": match["text"],
                "source": match["metadata"].get("type", "unknown"),
                "final_score": match.get("score", 0.0)
            })
        return chunks

    # optional debug helper to fetch raw recent items
    def fetch_recent(self, user_id: str, limit: int = 20):
        try:
            results = self.db.client.scroll(
                collection_name=self.db.user_history,
                scroll_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}}
                    ]
                },
                limit=limit
            )
            return results[0]
        except Exception:
            return []
