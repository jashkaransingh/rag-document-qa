"""
Embedding backends.

The default is sentence-transformers (all-MiniLM-L6-v2). It's small, fast, and
the embeddings are dense enough that retrieval quality holds up against much
larger models for typical document QA.

When the machine has no internet access to download model weights, the system
falls back to a TF-IDF based embedder. TF-IDF is not semantic, but it works
offline, runs in pure scikit-learn, and gives the rest of the pipeline a
sparse-vector signal to work with. Useful for air-gapped deployments and CI
environments where pulling 90MB of model weights is not appropriate.

Both backends expose the same interface so the rest of the system does not
care which one is in use.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of unit-normalized embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SentenceTransformerEmbedder(Embedder):
    """
    Default production embedder. Downloads a small model on first use and caches
    it locally. Embeddings are L2-normalized so we can use inner product as
    cosine similarity in FAISS.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False,
                                      convert_to_numpy=True)
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._model_name


class TfidfEmbedder(Embedder):
    """
    Offline fallback. Builds a TF-IDF vocabulary from a corpus, then projects
    queries into the same space. Vectors are dense (after toarray()) and
    L2-normalized.

    Limitations:
      - Cannot embed text outside the training vocabulary effectively
      - Sparse signal, not semantic
      - Must be fit before use, so the API is fit_then_embed instead of just embed
    """

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._max_features = max_features
        self._vectorizer = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True,
            ngram_range=(1, 2), stop_words="english", lowercase=True)
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Allow embed-only when caller is ok with fitting on this text
            self.fit(texts)
        sparse = self._vectorizer.transform(texts).astype(np.float32)
        dense = sparse.toarray()
        # L2-normalize each row
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)

    @property
    def dim(self) -> int:
        if not self._fitted:
            return self._max_features
        return len(self._vectorizer.vocabulary_)

    @property
    def name(self) -> str:
        return f"tfidf-{self.dim}d"


def build_embedder(prefer: str = "sentence-transformers",
                   fallback: bool = True) -> Embedder:
    """
    Factory that tries the preferred backend and falls back to TF-IDF if the
    preferred one cannot initialize (no internet, missing weights, etc).
    """
    if prefer == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            if not fallback:
                raise
            print(f"warning, sentence-transformers unavailable ({exc.__class__.__name__}), "
                  f"falling back to TF-IDF")
            return TfidfEmbedder()
    elif prefer == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedder preference: {prefer}")
