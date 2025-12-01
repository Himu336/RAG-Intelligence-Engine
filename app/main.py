# app/main.py

from fastapi import FastAPI, HTTPException
from app.schemas import RAGRequest, RAGResponse
from app.router import router as app_router
from app.text_interview.router import router as interview_router
from app.voice_interview.router import router as voice_interview_router

from app.vector_db.search_engine import VectorSearchEngine
from app.rag.prompt_builder import PromptBuilder
from app.llm.gemini_client import GeminiClient
from app.response.formatter import ResponseFormatter
from app.vector_db.user_history import UserHistoryManager
from app.vector_db.chat_memory import ChatMemory


# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------
app = FastAPI(
    title="AI Platform Service",
    description="Personal Coach + Interview Generator backend",
    version="1.0.0"
)

# Mount global/base router
app.include_router(app_router)

# Mount interview router
app.include_router(interview_router)

# Mount voice interview router
app.include_router(voice_interview_router)

# ----------------------------------------------------
# SERVICE INITIALIZATION
# ----------------------------------------------------
history_manager = UserHistoryManager()
chat_memory = ChatMemory(max_turns=6)
engine = VectorSearchEngine()
llm_client = GeminiClient()


# ----------------------------------------------------
# HELPER: Should we write long-term memory?
# ----------------------------------------------------
def _should_summarize(user_msg: str, ai_text: str) -> bool:
    """Decide whether to store long-term memory."""
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


# ----------------------------------------------------
# PERSONAL COACH / RAG ENDPOINT
# ----------------------------------------------------
@app.post("/rag", response_model=RAGResponse)
def run_rag(request: RAGRequest):
    user_id = request.user_id
    user_msg = (request.message or "").strip()

    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1) Add user turn to short-term memory
    chat_memory.add_user(user_id, user_msg)

    # 2) Get long-term memory from vector DB
    chunks = engine.search_relevant_chunks(query=user_msg, user_id=user_id)

    # 3) Get short-term memory window
    recent_turns = chat_memory.get_recent(user_id)

    # 4) Build LLM prompt
    prompt = PromptBuilder.build_prompt(
        user_query=user_msg,
        context_chunks=chunks,
        recent_conversation=recent_turns
    )

    # 5) Call LLM (Groq via GeminiClient shim)
    try:
        resp = llm_client.generate_raw(prompt)
    except Exception as e:
        ai_text = f"[LLM ERROR] {str(e)}"
        chat_memory.add_assistant(user_id, ai_text)
        return RAGResponse(ai_text=ai_text)

    # 6) Extract text (model-agnostic via GeminiClient shim)
    ai_text = llm_client.extract_text(resp)

    if not ai_text:
        ai_text = "[ERROR] No candidates returned or empty response."
        chat_memory.add_assistant(user_id, ai_text)
        return RAGResponse(ai_text=ai_text)

    # 7) Save assistant reply to short-term memory
    chat_memory.add_assistant(user_id, ai_text)

    # 8) Summarize into long-term memory (if meaningful)
    try:
        if _should_summarize(user_msg, ai_text):
            combined = f"User: {user_msg}\nAssistant: {ai_text}"
            facts = llm_client.summarize_to_facts(combined, max_facts=6)

            for f in (facts or []):
                f_clean = f.strip().strip('"').rstrip(",")
                if len(f_clean) < 8:
                    continue
                if f_clean.lower().startswith(("is named", "named ")):
                    continue

                history_manager.upsert_summary(user_id, f_clean)

    except Exception as e:
        print("⚠️ Summarization error:", e)

    # 9) Format response
    try:
        output = ResponseFormatter.format(ai_text)
    except Exception:
        output = {"ai_text": ai_text}

    return output
