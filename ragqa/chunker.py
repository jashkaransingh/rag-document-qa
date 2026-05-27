"""
Document chunking.

Wraps LangChain's RecursiveCharacterTextSplitter with the chunking strategy
we settled on after a lot of trial and error.

The defaults are 512 tokens (approximated as ~2048 characters at ~4 chars per
token) with 64-token overlap. Overlap matters because answers often live at
chunk boundaries, and without overlap you cut the model off mid-thought. Too
much overlap wastes embedding budget and inflates the index.

The separator priority order is intentional. Paragraphs split cleanly at \\n\\n,
sentences at single \\n or period, and only as a last resort do we split inside
a sentence. This keeps chunks semantically coherent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    source: str  # filename or url
    char_start: int
    char_end: int


class DocumentChunker:
    def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 256):
        """
        chunk_size and chunk_overlap are in characters, not tokens.
        2048 chars approximates 512 tokens for English.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def chunk_text(self, text: str, doc_id: str, source: str) -> List[Chunk]:
        """Split a single document into Chunks with byte offsets."""
        pieces = self.splitter.split_text(text)
        chunks = []
        cursor = 0
        for i, piece in enumerate(pieces):
            # Find this piece in the original text so we can store offsets
            idx = text.find(piece, cursor)
            if idx == -1:
                idx = cursor
            chunks.append(Chunk(
                text=piece,
                doc_id=doc_id,
                chunk_id=i,
                source=source,
                char_start=idx,
                char_end=idx + len(piece),
            ))
            cursor = max(idx + 1, cursor + 1)
        return chunks

    def chunk_file(self, path: str) -> List[Chunk]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        return self.chunk_text(text, doc_id=p.stem, source=p.name)
