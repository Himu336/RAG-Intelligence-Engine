# app/main.py
from fastapi import FastAPI
from app.schemas import RAGRequest, RAGResponse
from vector_db.search_engine import VectorSearchEngine
from rag.prompt_builder import PromptBuilder
from llm.gemini_client import GeminiClient
from response.formatter import ResponseFormatter
from vector_db.user_history import UserHistoryManager


app = FastAPI()

history_manager = UserHistoryManager()

    # save history
@app.post("/history/save")
def save_history(message: str, user_id: str):
    history_manager.save_message(user_id, message)
    return {"status": "saved"}

@app.post("/rag", response_model=RAGResponse)
def run_rag(request: RAGRequest):

    # 1. Get relevant chunks
    engine = VectorSearchEngine()
    chunks = engine.search_relevant_chunks(
        query=request.message,
        user_id=request.user_id
    )

    # 2. Build Prompt
    prompt = PromptBuilder.build_prompt(
        user_query=request.message,
        context_chunks=chunks
    )

    # 3. Call Gemini
    llm = GeminiClient()
    ai_output = llm.generate(prompt)

    # 4. Response formatting
    output = ResponseFormatter.format(ai_output)

    return output
