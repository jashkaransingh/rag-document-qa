# rag-document-qa

End-to-end Retrieval-Augmented Generation (RAG) system. Upload documents, ask questions, get grounded answers with citations. Multi-turn memory so follow-up questions work naturally.

## How it works

```
Document upload
      │
      ▼
Chunking (LangChain RecursiveTextSplitter)
      │
      ▼
Embedding (sentence-transformers)
      │
      ▼
FAISS vector index (persisted to disk)
      │
      ▼
Query → retrieve top-k chunks → inject into Gemini prompt
      │
      ▼
Answer with source citations + memory of prior turns
```

## API

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/ingest` | Upload and index a document (PDF or text) |
| POST | `/query` | Ask a question, get a grounded answer |
| DELETE | `/session/{id}` | Clear conversation memory |

## Quick start

```bash
git clone https://github.com/jashkaransingh/rag-document-qa
cd rag-document-qa
pip install -r requirements.txt
cp .env.example .env
python -m api.app
```

## What's interesting

- **Chunking strategy**: 512-token chunks with 64-token overlap — tuned to avoid splitting mid-sentence while keeping chunks semantically dense
- **Retrieval**: cosine similarity search over FAISS flat index, top-5 chunks reranked by MMR to reduce redundancy
- **Memory**: sliding window of last 6 turns injected into system prompt
- **Guardrails**: prompt injection detection via regex + keyword filter before hitting the LLM
