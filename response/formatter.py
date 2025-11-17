# response/formatter.py

class ResponseFormatter:
    @staticmethod
    def format(ai_text: str):
        # Add additional logic here later
        cleaned = ai_text.strip()
        return {"ai_text": cleaned}
