# rag/prompt_builder.py

class PromptBuilder:
    @staticmethod
    def build_prompt(user_query: str, context_chunks: list, recent_conversation: list):
        """
        Compose a compact prompt:
          - persona rules
          - top long-term memories (no scores)
          - recent conversation (oldest->newest) BUT don't duplicate the latest user message
          - the user's new message
          - response rules
        """

        # Long-term memory: keep up to 4 concise snippets and filter trivial ones
        filtered_chunks = []
        for c in (context_chunks or [])[:8]:
            text = (c.get("text") or "").strip()
            src = (c.get("source") or "memory").upper()
            # skip trivial very short items
            if len(text) < 10:
                continue
            # skip identity facts like "is named X"
            low = text.lower()
            if low.startswith("is named") or low.startswith("named "):
                continue
            filtered_chunks.append((src, text))

        if not filtered_chunks:
            context_text = "No long-term memories available."
        else:
            # include at most 4
            lines = []
            for src, txt in filtered_chunks[:4]:
                snippet = txt if len(txt) < 450 else txt[:450] + "..."
                lines.append(f"[{src}] {snippet}")
            context_text = "\n".join(lines)

        # Recent conversation: oldest -> newest, but don't duplicate last user message
        recent_lines = []
        if recent_conversation:
            recent_filtered = list(recent_conversation)
            # drop the very last recent turn if it's the same user message (dedupe)
            if recent_filtered:
                last = recent_filtered[-1]
                if last.get("role") == "user" and last.get("text", "").strip() == (user_query or "").strip():
                    recent_filtered = recent_filtered[:-1]

            for turn in recent_filtered:
                role = turn.get("role", "user")
                prefix = "User:" if role == "user" else "Assistant:"
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                # avoid repeating trivial lines
                if len(text) < 6:
                    continue
                recent_lines.append(f"{prefix} {text}")

        recent_text = "\n".join(recent_lines) if recent_lines else "No recent conversation."

        prompt = f"""You are a friendly personal coach. Help the user with planning, learning, productivity, career growth, and practical next steps.
Use the long-term memories and recent conversation below to personalize answers, but do not invent facts.

Long-term memory (relevant):
{context_text}

Recent conversation:
{recent_text}

User's new message:
{user_query}

Response rules:
- Keep replies short (4-7 short lines).
- Be practical, specific, and supportive.
- Give 1-3 actionable steps and 1 short follow-up question.
- Avoid medical, legal, or therapy-style advice.

Now reply as the user's personal coach in a short, helpful style."""
        return prompt.strip()
