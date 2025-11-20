# app/rag/rag_service.py

from typing import Dict, Any
from embeddings.generator import EmbeddingGenerator
from vector_db.search_engine import VectorSearchEngine
from rag.prompt_builder import PromptBuilder
from llm.gemini_client import GeminiClient
from rag.memory_extractor import MemoryExtractor
from vector_db.user_history import UserHistoryManager


class RAGService:
    """
    Standalone RAG engine (NOT used in main coach pipeline right now).
    Useful if you want a separate RAG endpoint later.
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.searcher = VectorSearchEngine()
        self.llm = GeminiClient()
        self.memory_mgr = UserHistoryManager()
        self.extractor = MemoryExtractor()

    def answer(self, user_id: str, user_message: str) -> Dict[str, Any]:
        # 1) Retrieve context
        context = self.searcher.search_relevant_chunks(
            query=user_message,
            user_id=user_id
        )

        # 2) Build prompt
        prompt = PromptBuilder.build_prompt(
            user_query=user_message,
            context_chunks=context,
            recent_conversation=[]  # standalone mode
        )

        # 3) Call LLM
        raw = self.llm.generate_raw(prompt)
        ai_text = self.llm.extract_text(raw)

        # 4) Heuristic memory extraction
        candidates = self.extractor.extract_candidates(user_message)
        if self.extractor.should_store(candidates):
            self.memory_mgr.save_message(user_id, user_message)

        # 5) Return structured result
        return {
            "answer": ai_text,
            "used_context": context,
            "stored_memory": candidates if self.extractor.should_store(candidates) else None,
        }
