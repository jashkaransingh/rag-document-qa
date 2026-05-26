# rag-document-qa

multi-turn RAG system. upload documents, ask questions, get grounded answers with citations. follow-up questions actually remember what you asked before.

## how it works

```
document upload
   │
   ▼
LangChain RecursiveTextSplitter (512 tokens, 64 overlap)
   │
   ▼
sentence-transformers embeddings
   │
   ▼
FAISS vector index (persisted to disk)
   │
   ▼
query → retrieve with MMR → inject into Gemini prompt
   │
   ▼
answer + source citations + memory of prior turns
```

## api

| method | route | what it does |
|--------|-------|-------------|
| POST | /ingest | upload and index a document (PDF or text) |
| POST | /query | ask a question, get a grounded answer |
| DELETE | /session/{id} | clear conversation memory |

## the hard part

retrieval quality. naive top-k kept returning the same chunk three times because everything in a doc clusters semantically close in embedding space. switched to maximum marginal relevance (MMR) to force diversity in retrieved chunks and answer quality jumped immediately. also spent an afternoon trying to break my own system with prompt injection and broke it faster than expected. guardrails came next, a mix of regex and keyword filters before queries hit the LLM. main lesson, the model is almost never the bottleneck. retrieval and chunking are.

## design decisions

- 512-token chunks with 64-token overlap to avoid splitting mid-sentence while keeping chunks semantically dense
- cosine similarity over a FAISS flat index, top-5 chunks reranked by MMR for diversity
- sliding window of the last 6 turns injected into the system prompt for multi-turn memory
- regex and keyword filter on incoming queries before they reach Gemini, catches the obvious injection patterns

## run it locally

```bash
git clone https://github.com/jashkaransingh/rag-document-qa
cd rag-document-qa
pip install -r requirements.txt
cp .env.example .env
python -m api.app
```

env vars

`GEMINI_API_KEY`, `EMBEDDING_MODEL`, `INDEX_PATH`

## stack

LangChain, sentence-transformers, FAISS, Gemini, Flask. all Python.
