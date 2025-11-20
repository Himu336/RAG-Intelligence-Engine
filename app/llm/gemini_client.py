# app/llm/gemini_client.py

import time
import json
import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable

from app.config import settings


# Configure API key once
genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiClient:
    """
    Lightweight + stable Gemini wrapper.

    Exposes:
      - generate_raw(prompt)
      - extract_text(response)
      - summarize_to_facts(text)
    """

    MODEL_NAME = "models/gemini-2.5-flash"

    def __init__(self):
        print(f"🧠 Using Gemini Model: {self.MODEL_NAME}")
        self.model = genai.GenerativeModel(self.MODEL_NAME)

    # ------------------------------------------------------------
    # RAW GENERATION (with retries)
    # ------------------------------------------------------------
    def generate_raw(self, prompt: str, max_output_tokens: int = 1024):
        last_exc = None

        for attempt in range(3):
            try:
                print("\n================ LLM PROMPT ================")
                print(prompt)
                print("============================================\n")

                print(f"🧠 Gemini call attempt {attempt + 1}")

                resp = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.6,
                        "max_output_tokens": max_output_tokens,
                    }
                )

                print("🔍 RAW GEMINI RESPONSE:", resp)
                return resp

            except Exception as e:
                last_exc = e
                msg = str(e).lower()

                # Retry only if model overloaded
                if "overloaded" in msg or "503" in msg or isinstance(e, ServiceUnavailable):
                    wait = (attempt + 1) * 2
                    print(f"⚠️ Model overloaded. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                print("❌ Non-retryable LLM error:", e)
                break

        raise Exception(f"[LLM ERROR] All attempts failed — last error: {last_exc}")

    # ------------------------------------------------------------
    # SAFE TEXT EXTRACTION
    # ------------------------------------------------------------
    def extract_text(self, resp):
        """Safely extract plain text from Gemini generate_content response."""
        try:
            candidate = resp.candidates[0]
            parts = getattr(candidate.content, "parts", [])

            if not parts:
                return ""

            text = "".join(
                part.text for part in parts
                if hasattr(part, "text") and part.text
            ).strip()

            return text

        except Exception:
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
