"""
Flask HTTP API for the RAG system.

Endpoints:
  POST /ingest       upload a text document (json {"text": ..., "source": ...})
  POST /ingest/file  multipart file upload
  POST /query        ask a question (json {"question": ..., "session_id": ...})
  DELETE /session/<id>   clear conversation memory for a session
  GET /healthz       liveness check
  GET /stats         basic numbers about the store and sessions

The system is instantiated once at module load with config read from
environment variables, so the same store is reused across requests. For
multi-process deployments, switch to a shared backing store (Redis, Postgres)
behind the same interface.
"""

import os
import uuid
from typing import Any, Dict

from flask import Flask, jsonify, request

from ragqa import RAGConfig, RAGSystem


def make_app(rag: RAGSystem) -> Flask:
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.get("/stats")
    def stats():
        return jsonify({
            "store_size": len(rag.store) if rag.store else 0,
            "embedder": rag.embedder.name,
            "embedding_dim": rag.embedder.dim,
            "llm": rag.llm.name,
            "sessions": rag.memory.session_count(),
            "retrieval": rag.config.retrieval,
            "k": rag.config.k,
        })

    @app.post("/ingest")
    def ingest():
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        if not text:
            return jsonify({"error": "missing 'text' field"}), 400
        source = body.get("source", "anonymous")
        doc_id = body.get("doc_id") or str(uuid.uuid4())[:8]
        n = rag.ingest_text(text, doc_id=doc_id, source=source)
        return jsonify({"chunks_added": n, "doc_id": doc_id,
                        "store_size": len(rag.store)})

    @app.post("/ingest/file")
    def ingest_file():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        f = request.files["file"]
        text = f.read().decode("utf-8", errors="ignore")
        source = f.filename or "uploaded"
        doc_id = os.path.splitext(source)[0]
        n = rag.ingest_text(text, doc_id=doc_id, source=source)
        return jsonify({"chunks_added": n, "doc_id": doc_id,
                        "store_size": len(rag.store)})

    @app.post("/query")
    def query():
        body = request.get_json(silent=True) or {}
        question = body.get("question")
        if not question:
            return jsonify({"error": "missing 'question' field"}), 400
        session_id = body.get("session_id", "default")
        k = body.get("k")
        result = rag.query(question, session_id=session_id, k=k)
        return jsonify({
            "answer": result.answer,
            "blocked": result.blocked,
            "block_reason": result.block_reason,
            "latency_ms": round(result.latency_ms, 2),
            "sources": [
                {"source": c.source, "doc_id": c.doc_id,
                 "chunk_id": c.chunk_id, "score": round(s, 4),
                 "preview": c.text[:160].replace("\n", " ")}
                for c, s in zip(result.sources, result.scores)
            ],
            "session_id": session_id,
        })

    @app.delete("/session/<session_id>")
    def clear_session(session_id):
        rag.memory.clear(session_id)
        return jsonify({"cleared": session_id})

    return app


def build_default_app() -> Flask:
    config = RAGConfig(
        embedder=os.environ.get("RAG_EMBEDDER", "sentence-transformers"),
        llm=os.environ.get("RAG_LLM", "anthropic"),
        retrieval=os.environ.get("RAG_RETRIEVAL", "mmr"),
        k=int(os.environ.get("RAG_K", "5")),
    )
    rag = RAGSystem(config=config)

    # Optional auto-ingest of a directory at startup
    auto_ingest = os.environ.get("RAG_AUTO_INGEST_DIR")
    if auto_ingest and os.path.isdir(auto_ingest):
        n = rag.ingest_directory(auto_ingest)
        print(f"auto-ingested {n} chunks from {auto_ingest}")

    return make_app(rag)


if __name__ == "__main__":
    app = build_default_app()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
