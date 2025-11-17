# rag/memory_extractor.py
from typing import Dict, Any
import re

class MemoryExtractor:
    """
    Lightweight memory extractor:
    - rule-based extraction of facts (goals, preferences, weaknesses, personal info)
    - can be extended to call an LLM-based extractor for richer memories
    """

    @staticmethod
    def extract_candidates(user_message: str) -> Dict[str, Any]:
        """
        Return dictionary of memory candidates. Keep them compact.
        Example keys: goal, preference, weakness, name, location
        """
        msg = user_message.strip()
        res = {}

        # simple heuristics
        # goal: "I want to learn X", "I want to become a X", "I want to get a job"
        goal_match = re.search(r"i (want|want to|would like to|wanna) (learn|become|get|build) (.+)", msg, re.I)
        if goal_match:
            res["goal"] = goal_match.group(3).strip()

        # weakness: "I'm weak in X", "I struggle with X"
        weak_match = re.search(r"(weak in|i'm weak in|i am weak in|struggle with|having trouble with) (.+)", msg, re.I)
        if weak_match:
            res["weakness"] = weak_match.group(2).strip()

        # preference: "I prefer", "I like"
        pref_match = re.search(r"(i prefer|i'd prefer|i like|i love) (.+)", msg, re.I)
        if pref_match:
            res["preference"] = pref_match.group(2).strip()

        # basic contact-like info (email or phone) - DO NOT SAVE unless user consent; we detect only
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", msg)
        if email_match:
            res["mention_email"] = email_match.group(0)

        return res

    @staticmethod
    def should_store(candidate_dict: Dict[str, Any]) -> bool:
        # store only if candidate has at least one high-value key
        return bool(candidate_dict.get("goal") or candidate_dict.get("weakness") or candidate_dict.get("preference"))
