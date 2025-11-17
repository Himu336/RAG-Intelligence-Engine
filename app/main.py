# app/main.py

from fastapi import FastAPI, HTTPException
from app.schemas import RAGRequest, RAGResponse
from vector_db.search_engine import VectorSearchEngine
from rag.prompt_builder import PromptBuilder
from llm.gemini_client import GeminiClient
from response.formatter import ResponseFormatter
from vector_db.user_history import UserHistoryManager
from vector_db.chat_memory import ChatMemory

app = FastAPI()

history_manager = UserHistoryManager()
chat_memory = ChatMemory(max_turns=6)
engine = VectorSearchEngine()
llm_client = GeminiClient()


def _should_summarize(user_msg: str, ai_text: str) -> bool:
    """Decide if we should extract long-term memories from this turn."""
    if not user_msg or len(user_msg.strip().split()) < 2:
        return False
    if len(user_msg.strip()) < 8:
        return False
    if not ai_text or len(ai_text.strip()) < 20:
        return False

    trivial = {"hi", "hello", "thanks", "thank you", "ok", "okay", "yes", "no"}
    if user_msg.strip().lower() in trivial:
        return False

    return True


@app.post("/rag", response_model=RAGResponse)
def run_rag(request: RAGRequest):
    user_id = request.user_id
    user_msg = (request.message or "").strip()

    # --- ✔ Block empty message ---
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # --- ✔ Add short-term memory turn ---
    chat_memory.add_user(user_id, user_msg)

    # --- ✔ Fetch relevant long-term memory ---
    chunks = engine.search_relevant_chunks(query=user_msg, user_id=user_id)

    # --- ✔ Get short-term context (last N turns) ---
    recent_turns = chat_memory.get_recent(user_id)

    # --- ✔ Build the coaching prompt ---
    prompt = PromptBuilder.build_prompt(
        user_query=user_msg,
        context_chunks=chunks,
        recent_conversation=recent_turns
    )

    # print(prompt)   # enable only when debugging

    # --- ✔ Call LLM safely ---
    try:
        resp = llm_client.generate_raw(prompt)
    except Exception as e:
        ai_text = f"[LLM ERROR] {str(e)}"
        chat_memory.add_assistant(user_id, ai_text)
        return RAGResponse(ai_text=ai_text)

    # --- ✔ Extract model output safely ---
    try:
        candidate = resp.candidates[0]
    except Exception:
        ai_text = "[ERROR] No candidates returned."
        chat_memory.add_assistant(user_id, ai_text)
        return RAGResponse(ai_text=ai_text)

    if not candidate.content or not getattr(candidate.content, "parts", []):
        safety = getattr(candidate, "safety_ratings", None)
        ai_text = f"[BLOCKED OR EMPTY RESPONSE] Safety: {safety}"
        chat_memory.add_assistant(user_id, ai_text)
        return RAGResponse(ai_text=ai_text)

    parts = candidate.content.parts or []
    ai_text = "".join(
        p.text for p in parts if hasattr(p, "text") and p.text
    ).strip() or "[LLM ERROR] empty text"

    # --- ✔ Save assistant message only short-term ---
    chat_memory.add_assistant(user_id, ai_text)

    # --- ✔ Summarize ONLY meaningful turns ---
    try:
        if _should_summarize(user_msg, ai_text):
            combined = f"User: {user_msg}\nAssistant: {ai_text}"
            facts = llm_client.summarize_to_facts(combined, max_facts=6)

            for f in (facts or []):
                f_clean = f.strip().strip('"').rstrip(",")

                if len(f_clean) < 8:
                    continue

                # skip identity
                low = f_clean.lower()
                if low.startswith("is named") or low.startswith("named "):
                    continue

                history_manager.upsert_summary(user_id, f_clean)

    except Exception as e:
        print("⚠️ Summarization/upsert failed:", e)

    # --- ✔ Format final reply ---
    try:
        output = ResponseFormatter.format(ai_text)
    except Exception:
        output = {"ai_text": ai_text}

    return output
