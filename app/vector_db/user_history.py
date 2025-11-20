# app/vector_db/user_history.py

import uuid
from typing import List, Dict, Any
import numpy as np

from app.vector_db.orm import VectorORM
from app.embeddings.generator import EmbeddingGenerator


# ------------------------- UTIL -------------------------

def _normalize_text(text: Any) -> str:
    """Ensure text is always a clean STRING (never list)."""
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join([str(x) for x in text])
    text = str(text).strip()
    text = text.strip("[]\"'")
    return text.strip()


def _is_trivial_text(s: str) -> bool:
    if not s:
        return True
    s = _normalize_text(s).lower()
    trivial = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "sure", "no", "yes"}
    if s in trivial:
        return True
    return len(s) < 10


# ----------------------- MANAGER ------------------------

class UserHistoryManager:
    """
    Long-term memory manager that:
    - Stores vectors + payloads
    - Uses scroll WITHOUT filter (manual filter)
    - Works on ALL Qdrant versions (no index required)
    """

    def __init__(self):
        self.db = VectorORM()
        self.emb = EmbeddingGenerator()
        self.max_summaries = 6

    # ------------------ RAW MESSAGE STORAGE ------------------

    def save_message(self, user_id: str, message: str, force: bool = False):
        message = _normalize_text(message)
        if not force and _is_trivial_text(message):
            return

        embedding = self.emb.create_embedding(message)

        self.db.insert(
            collection=self.db.user_history,
            text=message,
            embedding=embedding,
            metadata={
                "id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "type": "history",
                "vector": embedding,
            },
        )

    # ------------------- SUMMARY STORAGE ---------------------

    def save_summary(self, user_id: str, summary_text: str):
        summary_text = _normalize_text(summary_text)
        if not summary_text or _is_trivial_text(summary_text):
            return

        embedding = self.emb.create_embedding(summary_text)

        self.db.insert(
            collection=self.db.user_history,
            text=summary_text,
            embedding=embedding,
            metadata={
                "id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "type": "summary",
                "vector": embedding,
            },
        )

    # ------------------- SUMMARY FETCH -----------------------

    def get_summaries(self, user_id: str) -> List[Dict]:
        """Scroll all items then filter manually (NO FILTER —> NO ERROR)."""
        try:
            points, _ = self.db.client.scroll(
                collection_name=self.db.user_history,
                limit=500,          # enough for your use
                with_vectors=True,
            )
        except Exception as e:
            print("⚠️ ERROR get_summaries:", e)
            return []

        out = []

        for p in points:
            if p.payload.get("user_id") == str(user_id) and p.payload.get("type") == "summary":
                out.append({
                    "id": p.id,
                    "text": _normalize_text(p.payload.get("text", "")),
                    "vector": p.vector or p.payload.get("vector"),
                    "metadata": p.payload,
                })

        return out

    # ------------------- UPSERT SUMMARY ----------------------

    def upsert_summary(self, user_id: str, summary_text: str):
        summary_text = _normalize_text(summary_text)
        if not summary_text:
            return

        existing = self.get_summaries(user_id)
        normalized_new = " ".join(summary_text.lower().split())

        # check duplicates
        for e in existing:
            existing_norm = " ".join(e["text"].lower().split())
            if normalized_new == existing_norm:
                return
            if normalized_new in existing_norm:
                return

        # insert
        self.save_summary(user_id, summary_text)

        # enforce memory size
        summaries = self.get_summaries(user_id)
        if len(summaries) > self.max_summaries:
            to_delete = [s["id"] for s in summaries[self.max_summaries:]]
            for sid in to_delete:
                try:
                    self.db.delete(self.db.user_history, sid)
                except:
                    pass

    # ---------------- SEARCH RELEVANT ------------------------

    def search_relevant_chunks(self, query: str, user_id: str, limit: int = 5):
        summaries = self.get_summaries(user_id)
        if not summaries:
            return []

        query_emb = np.array(self.emb.create_embedding(query))
        scored = []

        for item in summaries:
            vec = item.get("vector")
            if vec is None:
                vec = self.emb.create_embedding(item["text"])

            vec = np.array(vec)

            score = float(
                np.dot(query_emb, vec) /
                (np.linalg.norm(query_emb) * np.linalg.norm(vec))
            )

            scored.append({
                "text": item["text"],
                "source": "summary",
                "final_score": score
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:limit]

    # ---------------------- DEBUG ---------------------------

    def fetch_recent(self, user_id: str, limit: int = 20):
        """Scroll everything and manually filter (safe)."""
        try:
            points, _ = self.db.client.scroll(
                collection_name=self.db.user_history,
                limit=500
            )
        except:
            return []

        return [
            p for p in points
            if p.payload.get("user_id") == str(user_id)
        ][:limit]
