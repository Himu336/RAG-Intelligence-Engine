# rag/rag_service.py
from embeddings.generator import EmbeddingGenerator
from vector_db.search_engine import VectorSearchEngine
from rag.prompt_builder import PromptBuilder
from llm.gemini_client import GeminiClient
from rag.memory_extractor import MemoryExtractor
from vector_db.user_history import UserHistoryManager
from typing import Dict, Any

class RAGService:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.searcher = VectorSearchEngine()
        self.llm = GeminiClient()
        self.memory_mgr = UserHistoryManager()
        self.extractor = MemoryExtractor()

    def answer(self, user_id: str, user_message: str) -> Dict[str, Any]:
        # 1) Retrieve best context chunks
        candidates = self.searcher.search_relevant_chunks(query=user_message, user_id=user_id)

        # 2) Build a prompt
        prompt = PromptBuilder.build_prompt(user_message, candidates)

        # 3) Call LLM
        llm_resp = self.llm.generate_text(prompt)

        # 4) Postprocess / Extract memories
        candidate_mem = self.extractor.extract_candidates(user_message)
        if self.extractor.should_store(candidate_mem):
            # store original message as memory (could store a summary instead)
            self.memory_mgr.save_message(user_id=user_id, message=user_message, memory_type="inference")
        else:
            # optionally store lightweight ephemeral logs (not saved as memory)
            pass

        # 5) Return structured result
        return {
            "answer": llm_resp,
            "used_context": candidates,
            "stored_memory": candidate_mem if self.extractor.should_store(candidate_mem) else None
        }
