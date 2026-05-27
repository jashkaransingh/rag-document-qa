import numpy as np
import pytest

from ragqa.chunker import Chunk
from ragqa.store import VectorStore


def make_chunks(texts):
    return [Chunk(text=t, doc_id="d", chunk_id=i, source="d.txt",
                  char_start=0, char_end=len(t))
            for i, t in enumerate(texts)]


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def test_add_and_search_topk():
    store = VectorStore(dim=4)
    embs = normalize(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    store.add(embs, make_chunks(["a", "b", "c"]))
    assert len(store) == 3

    query = normalize(np.array([[1, 0, 0, 0]], dtype=np.float32))
    results = store.search_topk(query[0], k=2)
    assert len(results) == 2
    assert results[0][0].text == "a"
    assert results[0][1] > results[1][1]


def test_dim_mismatch_raises():
    store = VectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add(np.zeros((1, 5), dtype=np.float32),
                  make_chunks(["x"]))


def test_mmr_returns_diverse_results():
    """MMR should prefer diverse chunks over highly similar ones."""
    store = VectorStore(dim=3)
    # Three chunks all highly similar (near [1, 0, 0]) and one diverse
    embs = normalize(np.array([
        [1.0, 0.05, 0.0],
        [1.0, 0.06, 0.0],
        [1.0, 0.07, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32))
    store.add(embs, make_chunks(["sim_a", "sim_b", "sim_c", "diverse"]))

    query = normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    # Top-k would return the 3 similar ones
    topk_results = store.search_topk(query, k=3)
    topk_texts = [c.text for c, _ in topk_results]
    assert topk_texts[:2] == ["sim_a", "sim_b"]

    # MMR should pull the diverse one into the result set
    mmr_results = store.search_mmr(query, k=3, fetch_k=4, lambda_mult=0.3)
    mmr_texts = [c.text for c, _ in mmr_results]
    assert "diverse" in mmr_texts


def test_save_and_load(tmp_path):
    store = VectorStore(dim=4)
    embs = normalize(np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=np.float32))
    store.add(embs, make_chunks(["a", "b"]))
    store.save(str(tmp_path))

    loaded = VectorStore.load(str(tmp_path))
    assert len(loaded) == 2
    assert loaded.chunks[0].text == "a"
    assert loaded.dim == 4

    query = normalize(np.array([[0, 1, 0, 0]], dtype=np.float32))[0]
    results = loaded.search_topk(query, k=1)
    assert results[0][0].text == "b"


def test_empty_store_search_returns_empty():
    store = VectorStore(dim=4)
    query = np.zeros(4, dtype=np.float32)
    assert store.search_topk(query, k=5) == []
    assert store.search_mmr(query, k=5) == []
