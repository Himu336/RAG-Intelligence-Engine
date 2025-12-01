# app/llm/gemini_client.py

import json
from typing import Any

from openai import OpenAI
from app.config import settings


class GeminiClient:
    """
    Backwards-compatible LLM wrapper that now uses OpenAI instead of Gemini.

    Exposes:
      - generate_raw(prompt)
      - extract_text(response)
      - summarize_to_facts(text)

    NOTE: We keep the class name `GeminiClient` so the rest of the codebase
    does not need to change imports, but under the hood this uses OpenAI.
    """

    # Groq LLaMA model for fast, high-quality chat
    MODEL_NAME = "llama-3.1-8b-instant"

    def __init__(self):
        print(f"🧠 Using Groq Model (GeminiClient shim): {self.MODEL_NAME}")
        # Use OpenAI client pointed at Groq's OpenAI-compatible endpoint
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        )

    # ------------------------------------------------------------
    # RAW GENERATION
    # ------------------------------------------------------------
    def generate_raw(self, prompt: str, max_output_tokens: int = 1024) -> Any:
        """
        Generate a raw OpenAI chat completion response.

        We keep the same signature and logging style as the old Gemini client.
        """
        try:
            print("\n================ LLM PROMPT ================")
            print(prompt)
            print("============================================\n")

            print("🧠 OpenAI chat.completions call")

            resp = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=max_output_tokens,
            )

            print("🔍 RAW OPENAI RESPONSE:", resp)
            return resp

        except Exception as e:
            print("❌ LLM error:", e)
            raise

    # ------------------------------------------------------------
    # SAFE TEXT EXTRACTION
    # ------------------------------------------------------------
    def extract_text(self, resp: Any) -> str:
        """
        Safely extract plain text from an OpenAI chat.completions response.

        This replaces the old Gemini-specific extraction logic.
        """
        try:
            # OpenAI chat completion response shape
            if hasattr(resp, "choices") and resp.choices:
                message = resp.choices[0].message
                content = getattr(message, "content", "") or ""
                return str(content).strip()

            # Fallback: try to treat resp as dict-like
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    message = choices[0].get("message", {})
                    return str(message.get("content", "")).strip()

        except Exception:
            pass

        return ""

    # ------------------------------------------------------------
    # SUMMARIZE TO SHORT FACTS (for long-term memory)
    # ------------------------------------------------------------
    def summarize_to_facts(self, text: str, max_facts: int = 8):
        if not text or len(text.strip()) < 10:
            return []

        prompt = f"""
Extract only *user facts* from the text below.
Return STRICT JSON: ["fact1", "fact2", ...]

Rules:
- Max {max_facts} items
- Remove opinions & assistant information
- Facts must be short phrases
- If none exist, return []

Text:
\"\"\"{text}\"\"\"
"""

        resp = self.generate_raw(prompt, max_output_tokens=512)
        out = self.extract_text(resp)

        if not out:
            return []

        # Attempt strict JSON parse
        try:
            if out.strip().startswith("["):
                arr = json.loads(out)
                if isinstance(arr, list):
                    return [str(f).strip() for f in arr if isinstance(f, str)][:max_facts]
        except Exception:
            pass

        # Fallback: treat output as bullets/lines
        facts = []
        for line in out.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if line:
                facts.append(line)
            if len(facts) >= max_facts:
                break

        return facts[:max_facts]
