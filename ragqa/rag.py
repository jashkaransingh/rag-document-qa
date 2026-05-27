"""
RAG orchestrator.

This is the top-level object you instantiate to use the system. It owns the
chunker, embedder, vector store, memory, guardrails, and LLM. Everything below
it is a leaf component.

Two main operations:
  - ingest(path or text), chunks the document, embeds the chunks, adds to the
    store
  - query(question, session_id), runs guardrails, retrieves with MMR, builds
    the prompt with memory + context, calls the LLM, stores the turn in memory
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from .chunker import Chunk, DocumentChunker
from .embedder import Embedder, TfidfEmbedder, build_embedder
from .guardrails import GuardrailResult, Guardrails
from .llm import LLMClient, LLMResponse, build_llm, build_prompt
from .memory import ConversationMemory
from .store import VectorStore


@dataclass
class QueryResult:
    answer: str
    sources: List[Chunk]
    scores: List[float]
    blocked: bool = False
    block_reason: str = ""
    latency_ms: float = 0.0
    llm_response: Optional[LLMResponse] = None


@dataclass
class RAGConfig:
    chunk_size: int = 2048
    chunk_overlap: int = 256
    embedder: str = "sentence-transformers"  # "sentence-transformers" or "tfidf"
    llm: str = "anthropic"  # "anthropic" or "stub"
    retrieval: Literal["topk", "mmr"] = "mmr"
    k: int = 5
    fetch_k: int = 20
    mmr_lambda: float = 0.5
    memory_window: int = 6
    guardrail_threshold: float = 0.9


class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None,
                 embedder: Optional[Embedder] = None,
                 llm: Optional[LLMClient] = None):
        self.config = config or RAGConfig()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap)
        self.embedder = embedder or build_embedder(self.config.embedder)
        self.store: Optional[VectorStore] = None  # created lazily on first ingest
        self.memory = ConversationMemory(window_turns=self.config.memory_window)
        self.guardrails = Guardrails(threshold=self.config.guardrail_threshold)
        self.llm = llm or build_llm(self.config.llm)

    # ---- ingestion ---------------------------------------------------------

    def _ensure_store(self) -> VectorStore:
        if self.store is None:
            self.store = VectorStore(dim=self.embedder.dim)
        return self.store

    def ingest_text(self, text: str, doc_id: Optional[str] = None,
                    source: Optional[str] = None) -> int:
        """Chunk, embed, store. Returns the number of chunks added."""
        doc_id = doc_id or str(uuid.uuid4())[:8]
        source = source or doc_id
        chunks = self.chunker.chunk_text(text, doc_id=doc_id, source=source)
        return self._add_chunks(chunks)

    def ingest_file(self, path: str) -> int:
        chunks = self.chunker.chunk_file(path)
        return self._add_chunks(chunks)

    def ingest_directory(self, directory: str,
                         extensions: Tuple[str, ...] = (".txt", ".md")) -> int:
        total = 0
        for p in sorted(Path(directory).rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                total += self.ingest_file(str(p))
        return total

    def _add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        # Special case for TF-IDF: vocabulary changes when new docs come in, so
        # we re-fit on the combined corpus and rebuild the store from scratch
        # with the new dimension. This is the only embedder that needs this.
        if isinstance(self.embedder, TfidfEmbedder):
            existing_chunks = list(self.store.chunks) if self.store else []
            all_chunks = existing_chunks + chunks
            all_text = [c.text for c in all_chunks]

            # Re-fit on the full corpus
            self.embedder._fitted = False  # force a fresh fit
            self.embedder.fit(all_text)
            embeddings = self.embedder.embed(all_text)

            # Build a new store at the new dim
            self.store = VectorStore(dim=self.embedder.dim)
            self.store.add(embeddings, all_chunks)
            return len(chunks)

        # Normal (dense) embedder path
        store = self._ensure_store()
        embeddings = self.embedder.embed([c.text for c in chunks])
        store.add(embeddings, chunks)
        return len(chunks)

    # ---- query -------------------------------------------------------------

    def query(self, question: str, session_id: str = "default",
              k: Optional[int] = None) -> QueryResult:
        start = time.perf_counter()

        gr = self.guardrails.check(question)
        if not gr.allowed:
            return QueryResult(
                answer=gr.message, sources=[], scores=[],
                blocked=True, block_reason=", ".join(gr.reasons),
                latency_ms=(time.perf_counter() - start) * 1000)

        if self.store is None or len(self.store) == 0:
            return QueryResult(
                answer="No documents have been ingested yet.",
                sources=[], scores=[], latency_ms=(time.perf_counter() - start) * 1000)

        k = k or self.config.k
        query_emb = self.embedder.embed([question])[0]
        if self.config.retrieval == "mmr":
            retrieved = self.store.search_mmr(
                query_emb, k=k, fetch_k=self.config.fetch_k,
                lambda_mult=self.config.mmr_lambda)
        else:
            retrieved = self.store.search_topk(query_emb, k=k)

        chunks = [c for c, _ in retrieved]
        scores = [s for _, s in retrieved]

        history = self.memory.get_history(session_id)
        system, messages = build_prompt(question, chunks, history)
        llm_resp = self.llm.generate(system, messages)

        self.memory.add_user(session_id, question)
        self.memory.add_assistant(session_id, llm_resp.text)

        return QueryResult(
            answer=llm_resp.text,
            sources=chunks,
            scores=scores,
            latency_ms=(time.perf_counter() - start) * 1000,
            llm_response=llm_resp,
        )

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        if self.store is None:
            raise RuntimeError("nothing to save, no ingestion has happened")
        self.store.save(path)

    def load(self, path: str) -> None:
        self.store = VectorStore.load(path)
