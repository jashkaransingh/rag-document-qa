"""
FAISS vector store with persistence and Maximum Marginal Relevance retrieval.

This is the half of the system that took the most time to get right. Naive
top-k retrieval kept returning chunks that were all semantically clustered, so
the LLM was being asked to answer based on three copies of the same idea
rather than a diverse view of the document. Switching to MMR forced the
retrieval set to balance relevance to the query against diversity from chunks
already picked, and answer quality jumped.

MMR works like this. From the top N candidates by cosine similarity, pick the
one with the highest relevance score, then iteratively pick the next chunk
that maximizes:
    lambda * sim(chunk, query) - (1 - lambda) * max sim(chunk, already_picked)
Higher lambda biases toward relevance, lower biases toward diversity. Default
0.5 is balanced.

Persistence writes the FAISS index and the chunk metadata to disk side by side
so the system can be restarted without re-embedding everything.
"""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # IndexFlatIP works with normalized embeddings = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {embeddings.shape[1]} != store dim {self.dim}")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings count mismatch")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def search_topk(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Plain top-k by inner product (cosine on normalized vectors)."""
        if len(self.chunks) == 0:
            return []
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        scores, indices = self.index.search(
            query_emb.astype(np.float32), min(k, len(self.chunks)))
        return [(self.chunks[idx], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx != -1]

    def search_mmr(self, query_emb: np.ndarray, k: int = 5,
                   fetch_k: int = 20, lambda_mult: float = 0.5
                   ) -> List[Tuple[Chunk, float]]:
        """
        Maximum Marginal Relevance retrieval.

        Pull fetch_k by similarity first, then iteratively pick k chunks that
        balance relevance and diversity.
        """
        if len(self.chunks) == 0:
            return []
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]

        fetch_k = min(fetch_k, len(self.chunks))
        scores, indices = self.index.search(
            query_emb.astype(np.float32), fetch_k)
        candidate_idx = [int(i) for i in indices[0] if i != -1]
        candidate_scores = list(scores[0][:len(candidate_idx)])

        # Recover candidate embeddings from FAISS for diversity computation
        candidate_embs = np.vstack([
            self.index.reconstruct(int(i)) for i in candidate_idx
        ])

        selected: List[int] = []
        selected_embs: List[np.ndarray] = []

        # First pick is just the most relevant one
        if candidate_idx:
            selected.append(0)
            selected_embs.append(candidate_embs[0])

        while len(selected) < k and len(selected) < len(candidate_idx):
            best_score = -np.inf
            best_idx = -1
            for ci, emb in enumerate(candidate_embs):
                if ci in selected:
                    continue
                relevance = float(candidate_scores[ci])
                if selected_embs:
                    sims_to_selected = [
                        float(np.dot(emb, sel)) for sel in selected_embs]
                    diversity_penalty = max(sims_to_selected)
                else:
                    diversity_penalty = 0.0
                mmr_score = (lambda_mult * relevance
                             - (1.0 - lambda_mult) * diversity_penalty)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = ci
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])

        return [(self.chunks[candidate_idx[i]],
                 float(candidate_scores[i])) for i in selected]

    def save(self, dir_path: str) -> None:
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(out / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        in_path = Path(dir_path)
        with open(in_path / "meta.json") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(in_path / "index.faiss"))
        with open(in_path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
