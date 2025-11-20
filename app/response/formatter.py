# app/response/formatter.py

class ResponseFormatter:
    """
    Central place to format responses before sending them back to the client.
    Keeps your output consistent and allows future extension
    (e.g., markdown cleanup, trimming, emojis, safety).
    """

    @staticmethod
    def format(ai_text: str) -> dict:
        if not isinstance(ai_text, str):
            ai_text = str(ai_text or "")

        cleaned = ai_text.strip()

        # Future improvements:
        # - strip markdown artifacts
        # - limit extremely long outputs
        # - unify line breaks
        # - optional emoji polishing

        return {"ai_text": cleaned}
