from ragqa.chunker import DocumentChunker


def test_chunks_have_offsets():
    text = "This is paragraph one.\n\nThis is paragraph two with more text."
    chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_text(text, doc_id="doc1", source="doc1.txt")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc1"
        assert c.source == "doc1.txt"
        assert c.char_start >= 0
        assert c.char_end > c.char_start


def test_chunk_ids_are_sequential():
    text = "a " * 500
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text(text, "d", "d.txt")
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_short_text_one_chunk():
    text = "short text"
    chunker = DocumentChunker(chunk_size=1000)
    chunks = chunker.chunk_text(text, "d", "d.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
