# llm/gemini_client.py
import time
import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable
from app.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiClient:
    """
    Minimal, dependable Gemini client using only models/gemini-2.5-flash.
    Exposes:
      - generate_raw(prompt) -> raw generate_content response
      - summarize_to_facts(text) -> list[str] (compact facts for long-term memory)
    """

    MODEL_NAME = "models/gemini-2.5-flash"

    def __init__(self):
        print("🧠 Using Gemini model:", self.MODEL_NAME)
        self.model = genai.GenerativeModel(self.MODEL_NAME)

    def generate_raw(self, prompt: str, max_output_tokens: int = 1024):
        last_exc = None
        for attempt in range(3):
            try:

                print("\n================ LLM PROMPT SENT TO GEMINI ================")
                print(prompt)
                print("===========================================================\n")
                print(f"🧠 Calling Gemini 2.5-flash (attempt {attempt+1})")
                resp = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.6,
                        "max_output_tokens": max_output_tokens
                    }
                )
                print("🔍 RAW GEMINI RESPONSE:", resp)
                return resp
            except Exception as e:
                last_exc = e
                s = str(e).lower()
                if "overloaded" in s or "503" in s or isinstance(e, ServiceUnavailable):
                    wait = (attempt + 1) * 2
                    print(f"⚠️ Model overloaded. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                print("❌ Non-retryable LLM error:", e)
                break
        raise Exception(f"[LLM ERROR] All model attempts failed. Last error: {last_exc}")

    def _extract_text_from_response(self, resp):
        # Safe extraction of text from generate_content response
        try:
            cand = resp.candidates[0]
            if not cand.content or not getattr(cand.content, "parts", []):
                return ""
            parts = cand.content.parts or []
            extracted = "".join([p.text for p in parts if hasattr(p, "text") and p.text]).strip()
            return extracted
        except Exception:
            return ""

    def summarize_to_facts(self, text: str, max_facts: int = 8):
        """
        Ask the LLM to extract short, factual bullet points about the USER from the given text.
        Returns a list of short strings (facts). If no facts found, returns [].
        """
        if not text or len(text.strip()) < 10:
            return []

        prompt = f"""
Extract only user facts from the following conversation text. Output a JSON array of short bullet facts (no more than {max_facts} items).
- Each fact should be one short sentence or phrase.
- Do NOT include assistant text, opinions, or anything uncertain.
- Examples of facts: "Prefers Node.js", "Has built 3 Node.js projects", "Goal: become backend developer", "Learns best with short tasks".
- If there are no user facts, return an empty JSON array: [].

Text:
\"\"\"{text}\"\"\"
"""

        resp = self.generate_raw(prompt, max_output_tokens=512)
        out = self._extract_text_from_response(resp)
        if not out:
            return []

        # Attempt to parse JSON-like output or split lines/bullets as fallback
        facts = []
        # naive parse if the model returned a JSON array string
        try:
            import json
            maybe_json = out.strip()
            if maybe_json.startswith("["):
                parsed = json.loads(maybe_json)
                if isinstance(parsed, list):
                    facts = [f.strip() for f in parsed if isinstance(f, str) and f.strip()]
                    return facts[:max_facts]
        except Exception:
            pass

        # Fallback: split by lines and bullets
        for line in out.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if line:
                facts.append(line)
            if len(facts) >= max_facts:
                break

        return facts[:max_facts]
