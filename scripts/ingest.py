"""CLI to ingest documents into a persistent store."""

import argparse
import os

from ragqa import RAGConfig, RAGSystem


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG store")
    parser.add_argument("path", help="file or directory to ingest")
    parser.add_argument("--store-dir", default="store",
                        help="where to save the FAISS index and chunk metadata")
    parser.add_argument("--embedder", default="sentence-transformers",
                        choices=["sentence-transformers", "tfidf"])
    parser.add_argument("--extensions", nargs="+",
                        default=[".txt", ".md"],
                        help="file extensions to include when path is a directory")
    args = parser.parse_args()

    cfg = RAGConfig(embedder=args.embedder, llm="stub")  # llm unused for ingest
    rag = RAGSystem(config=cfg)

    if os.path.isdir(args.path):
        n = rag.ingest_directory(args.path,
                                  extensions=tuple(args.extensions))
    else:
        n = rag.ingest_file(args.path)

    rag.save(args.store_dir)
    print(f"ingested {n} chunks, saved to {args.store_dir}")


if __name__ == "__main__":
    main()
