# app/rag/memory_extractor.py

import re
from typing import Dict, Any


class MemoryExtractor:
    """
    Lightweight heuristic memory extractor.
    Extracts possible long-term user info:
      - goals
      - weaknesses
      - preferences
    """

    @staticmethod
    def extract_candidates(user_message: str) -> Dict[str, Any]:
        msg = user_message.strip()
        res = {}

        # Goals: “I want to learn X”
        goal_match = re.search(
            r"i (want|want to|would like to|wanna) (learn|become|get|build) (.+)",
            msg, re.I
        )
        if goal_match:
            res["goal"] = goal_match.group(3).strip()

        # Weakness: “I'm weak in X”
        weak_match = re.search(
            r"(weak in|i'm weak in|i am weak in|struggle with|having trouble with) (.+)",
            msg, re.I
        )
        if weak_match:
            res["weakness"] = weak_match.group(2).strip()

        # Preferences: “I prefer X”
        pref_match = re.search(
            r"(i prefer|i'd prefer|i like|i love) (.+)",
            msg, re.I
        )
        if pref_match:
            res["preference"] = pref_match.group(2).strip()

        # Email detection (only detect, never store)
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", msg)
        if email_match:
            res["mention_email"] = email_match.group(0)

        return res

    @staticmethod
    def should_store(candidates: Dict[str, Any]) -> bool:
        """Store only if high-value items exist."""
        return bool(
            candidates.get("goal")
            or candidates.get("weakness")
            or candidates.get("preference")
        )
