# app/vector_db/chat_memory.py

import json
import redis
from typing import List, Dict
from app.config import settings


class ChatMemory:
    """
    Short-term conversation memory stored in Redis with TTL.
    Uses Redis Cloud (TLS-compatible).
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns

        try:
            self.r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                ssl_cert_reqs=None,        # Works for Redis Cloud TLS
                decode_responses=True
            )
            self.r.ping()
            print("🔌 Connected to Redis Cloud!")
        except Exception as e:
            print("❌ Redis connection error:", e)
            raise e

        self.ttl_seconds = getattr(settings, "CHAT_TTL_SECONDS", 3600)

    def _key(self, user_id: str):
        return f"chat_memory:{user_id}"

    def _push(self, user_id: str, role: str, text: str):
        key = self._key(user_id)
        data = json.dumps({"role": role, "text": text})

        try:
            self.r.lpush(key, data)
            self.r.ltrim(key, 0, self.max_turns - 1)
            self.r.expire(key, self.ttl_seconds)
        except Exception as e:
            print("❌ Redis write failed:", e)

    def add_user(self, user_id: str, message: str):
        self._push(user_id, "user", message)

    def add_assistant(self, user_id: str, message: str):
        self._push(user_id, "assistant", message)

    def get_recent(self, user_id: str) -> List[Dict]:
        key = self._key(user_id)

        try:
            raw = self.r.lrange(key, 0, self.max_turns - 1)
        except Exception as e:
            print("❌ Redis read failed:", e)
            return []

        out = []
        for item in reversed(raw):
            try:
                out.append(json.loads(item))
            except:
                pass

        return out

    def clear(self, user_id: str):
        try:
            self.r.delete(self._key(user_id))
        except Exception as e:
            print("❌ Redis delete failed:", e)
