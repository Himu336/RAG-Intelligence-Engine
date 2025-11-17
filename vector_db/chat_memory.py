# vector_db/chat_memory.py

import json
import redis
from typing import List, Dict
from datetime import timedelta
from app.config import settings


class ChatMemory:
    """
    Short-term conversation memory stored in Redis with TTL.
    - Keeps last N turns
    - Each user_id has a Redis LIST that expires after TTL
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns

        # Redis Cloud connection (TLS compatible)
        # Note: Some redis-py versions DO NOT support ssl=True.
        # Redis Cloud works using ssl_cert_reqs=None for TLS.
        try:
            self.r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                ssl_cert_reqs=None,     # IMPORTANT FIX: Works for Redis Cloud
                decode_responses=True
            )

            # Test connection once
            self.r.ping()
            print("🔌 Connected to Redis Cloud successfully!")
        except Exception as e:
            print("❌ Redis connection failed:", e)
            raise e

        # TTL for short-term memory
        self.ttl_seconds = getattr(settings, "CHAT_TTL_SECONDS", 3600)  # Default: 1 hour

    # Internal helper to construct Redis key
    def _key(self, user_id: str):
        return f"chat_memory:{user_id}"

    def _push(self, user_id: str, role: str, text: str):
        """Push one item into Redis & enforce TTL + max list length."""
        key = self._key(user_id)
        entry = json.dumps({"role": role, "text": text})

        try:
            # LPUSH → newest at index 0
            self.r.lpush(key, entry)

            # Keep only max_turns items
            self.r.ltrim(key, 0, self.max_turns - 1)

            # Refresh TTL
            self.r.expire(key, self.ttl_seconds)

        except Exception as e:
            print("❌ Redis write failed:", e)

    # === PUBLIC API ===

    def add_user(self, user_id: str, message: str):
        self._push(user_id, "user", message)

    def add_assistant(self, user_id: str, message: str):
        self._push(user_id, "assistant", message)

    def get_recent(self, user_id: str) -> List[Dict]:
        """Return messages in correct chronological order (oldest → newest)."""
        key = self._key(user_id)

        try:
            # Because LPUSH inserts newest first, LRANGE returns newest first.
            raw_list = self.r.lrange(key, 0, self.max_turns - 1)
        except Exception as e:
            print("❌ Redis read failed:", e)
            return []

        result = []
        for x in reversed(raw_list):  # Reverse to get oldest → newest
            try:
                result.append(json.loads(x))
            except json.JSONDecodeError:
                pass

        return result

    def clear(self, user_id: str):
        """Manually clear short-term memory for a user."""
        try:
            self.r.delete(self._key(user_id))
        except Exception as e:
            print("❌ Redis delete failed:", e)
