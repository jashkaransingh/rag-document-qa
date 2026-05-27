import numpy as np

from ragqa.embedder import TfidfEmbedder, build_embedder


def test_tfidf_produces_normalized_vectors():
    emb = TfidfEmbedder(max_features=128)
    emb.fit(["hello world", "machine learning is fun",
             "the quick brown fox"])
    vectors = emb.embed(["hello world"])
    assert vectors.shape == (1, emb.dim)
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_tfidf_distinct_inputs_get_distinct_vectors():
    emb = TfidfEmbedder(max_features=128)
    corpus = ["hello world", "completely unrelated text about pasta",
              "another sentence here"]
    emb.fit(corpus)
    a, b = emb.embed(["hello world", "completely unrelated text about pasta"])
    similarity = float(np.dot(a, b))
    assert similarity < 0.5


def test_factory_falls_back_to_tfidf_when_st_unavailable(monkeypatch):
    # Force the sentence-transformers path to fail and verify we get tfidf
    import ragqa.embedder as emb_mod

    class BrokenST:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated no internet")

    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", BrokenST)
    embedder = build_embedder(prefer="sentence-transformers", fallback=True)
    assert "tfidf" in embedder.name


def test_factory_raises_without_fallback(monkeypatch):
    import ragqa.embedder as emb_mod
    import pytest

    class BrokenST:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated no internet")

    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", BrokenST)
    with pytest.raises(RuntimeError):
        build_embedder(prefer="sentence-transformers", fallback=False)
