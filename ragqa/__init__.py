"""ragqa, a small RAG system for document QA with multi-turn memory."""

from .chunker import Chunk, DocumentChunker
from .embedder import (Embedder, SentenceTransformerEmbedder, TfidfEmbedder,
                       build_embedder)
from .guardrails import GuardrailResult, Guardrails
from .llm import (AnthropicLLM, LLMClient, LLMResponse, StubLLM, build_llm,
                  build_prompt)
from .memory import ConversationMemory, Turn
from .rag import QueryResult, RAGConfig, RAGSystem
from .store import VectorStore

__all__ = [
    "Chunk", "DocumentChunker",
    "Embedder", "SentenceTransformerEmbedder", "TfidfEmbedder",
    "build_embedder",
    "GuardrailResult", "Guardrails",
    "AnthropicLLM", "LLMClient", "LLMResponse", "StubLLM",
    "build_llm", "build_prompt",
    "ConversationMemory", "Turn",
    "QueryResult", "RAGConfig", "RAGSystem",
    "VectorStore",
]

__version__ = "0.1.0"
