# app/rag/prompt_builder.py

class PromptBuilder:
    @staticmethod
    def build_prompt(user_query: str, context_chunks: list, recent_conversation: list):
        """
        Build coaching prompt using:
          - relevant long-term memory
          - short-term memory (recent turns)
          - current user message
        """
        # ------ Long-term memory ------
        filtered_chunks = []
        for c in (context_chunks or [])[:8]:
            txt = (c.get("text") or "").strip()
            src = (c.get("source") or "memory").upper()

            if len(txt) < 10:
                continue
            if txt.lower().startswith(("is named", "named ")):
                continue

            snippet = txt if len(txt) < 450 else txt[:450] + "..."
            filtered_chunks.append(f"[{src}] {snippet}")

        context_text = (
            "\n".join(filtered_chunks[:4])
            if filtered_chunks else
            "No long-term memories available."
        )

        # ------ Recent conversation ------
        recent_lines = []
        if recent_conversation:
            turns = list(recent_conversation)

            if turns:
                last = turns[-1]
                if last.get("role") == "user" and last.get("text", "").strip() == user_query.strip():
                    turns = turns[:-1]

            for t in turns:
                txt = (t.get("text") or "").strip()
                if len(txt) < 6:
                    continue
                prefix = "User:" if t.get("role") == "user" else "Assistant:"
                recent_lines.append(f"{prefix} {txt}")

        recent_text = (
            "\n".join(recent_lines)
            if recent_lines else
            "No recent conversation."
        )

        # ------ Final prompt ------
        prompt = f"""
You are a friendly personal coach. Help the user with planning, learning, productivity, and career growth, practical next steps.
Use the long-term memories and recent conversation below to personalize answers, but do not invent facts.

Long-term memory:
{context_text}

Recent conversation:
{recent_text}

User's new message:
{user_query}

Rules:
- Reply in 4–7 short lines.
- Keep tone supportive and practical.
- Give 1-3 actionable steps and 1 short follow-up question. if needed for clarity.

Now respond as the user's personal coach.
"""
        return prompt.strip()
