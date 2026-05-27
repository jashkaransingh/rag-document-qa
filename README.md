# rag-document-qa

multi-turn document QA system. ingest docs, ask questions, get grounded answers with source citations. follow-up questions remember context from earlier in the conversation. prompt injection attempts get filtered before they reach the LLM.

```
$ python3 scripts/demo.py --llm anthropic
ingested 6 chunks from data/sample_docs
Q: How heavy is the Mark VII and what's its payload capacity?
A: The Mark VII has a mass of 142 kilograms and supports a continuous
   payload capacity of 120 kg (180 kg peak). [1]
   sources: [1] mark_vii_specs.md  (score 0.62)
   latency: 412 ms
```

## architecture

```
document upload
   │
   ▼  langchain RecursiveCharacterTextSplitter (2048 chars, 256 overlap)
chunks with source + offset metadata
   │
   ▼  sentence-transformers (or tfidf fallback)
L2-normalized embeddings
   │
   ▼  faiss IndexFlatIP, cosine similarity
vector store with disk persistence
   │
   ▼  query → guardrails → MMR retrieval (k=5 from top 20)
retrieved chunks + sliding-window conversation history
   │
   ▼  anthropic Claude (or deterministic stub for offline use)
grounded answer with [n] citations + memory of prior turns
```

## endpoints

| method | route | what it does |
|--------|-------|-------------|
| POST | /ingest | upload a text document by json body |
| POST | /ingest/file | upload a document by multipart file |
| POST | /query | ask a question, get a grounded answer |
| DELETE | /session/{id} | clear conversation memory for a session |
| GET | /healthz | liveness check |
| GET | /stats | store size, embedder, sessions, retrieval config |

## the hard part

three things, in order of how much pain they caused.

**retrieval quality.** naive top-k kept returning the same chunk three times because everything in a doc clusters semantically close in embedding space. swapping to maximum marginal relevance forced diversity in the retrieved set and answer quality jumped immediately. MMR pulls fetch_k candidates by similarity, then iteratively picks k chunks that maximize `lambda * relevance - (1 - lambda) * max_similarity_to_already_picked`. lambda 0.5 is the balanced default, lower biases toward diversity. lesson that stuck with me, the model is rarely the bottleneck, the retrieval pipeline is.

**prompt injection.** spent an afternoon trying to break my own system and broke it faster than expected. role overrides, system prompt extraction, persona hijacks, delimiter confusion. wrote `ragqa/guardrails.py` with a weighted regex set covering the common attack categories, scored each query before it hits retrieval, blocked anything over threshold with a generic message. tuned the patterns conservatively, false positives are recoverable (rephrase), false negatives are not.

**embedder fallback.** sentence-transformers needs to download model weights from huggingface, which is fine in production but breaks CI, air-gapped environments, and anyone trying the project on a machine without internet. wrote a TF-IDF embedder that implements the same `Embedder` interface and gets auto-selected when the dense backend cannot initialize. not semantic, but it keeps the pipeline functional and demonstrable everywhere.

## design decisions

- 2048-char chunks with 256 overlap (~512 tokens / 64 tokens) to avoid splitting mid-sentence while keeping chunks semantically dense
- cosine similarity over a FAISS flat index, top-5 chunks reranked by MMR with lambda 0.5
- sliding window of the last 6 turns injected into the LLM prompt for multi-turn memory
- regex-based guardrails before retrieval, conservative threshold tuned so generic questions never trip
- pluggable LLM and embedder backends behind small interfaces so you can swap in whatever you want

## quickstart

```bash
git clone https://github.com/jashkaransingh/rag-document-qa
cd rag-document-qa
pip install -r requirements.txt
cp .env.example .env
# put your ANTHROPIC_API_KEY in .env

# run the demo against the sample docs
python3 scripts/demo.py --llm anthropic

# or no api key, run with the offline stub backend
python3 scripts/demo.py --embedder tfidf --llm stub
```

## use as a library

```python
from ragqa import RAGSystem, RAGConfig

rag = RAGSystem(config=RAGConfig(retrieval="mmr", k=5))

rag.ingest_file("docs/policy.md")
rag.ingest_directory("docs/", extensions=(".md", ".txt"))

result = rag.query("What's the refund policy?", session_id="user-123")
print(result.answer)
for c, score in zip(result.sources, result.scores):
    print(f"  {c.source} (score {score:.2f})")
```

## use as an HTTP service

```bash
ANTHROPIC_API_KEY=sk-ant-... \
RAG_AUTO_INGEST_DIR=data/sample_docs \
python3 -m api.app
# listening on port 8000

curl -X POST localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"question": "what is the payload capacity?", "session_id": "s1"}'
```

## project layout

```
rag-document-qa/
├── ragqa/
│   ├── chunker.py        langchain RecursiveCharacterTextSplitter wrapper
│   ├── embedder.py       sentence-transformers + tfidf fallback
│   ├── store.py          faiss store with MMR retrieval and persistence
│   ├── memory.py         sliding window conversation memory
│   ├── guardrails.py     prompt injection filter
│   ├── llm.py            anthropic client + stub for offline use
│   └── rag.py            orchestrator that ties everything together
├── api/
│   └── app.py            flask HTTP API
├── scripts/
│   ├── ingest.py         CLI to ingest a directory into a persistent store
│   └── demo.py           end-to-end transcript runner
├── tests/                pytest suite, 37 tests
├── data/sample_docs/     example documents for the demo
└── samples/              recorded demo output
```

## demo transcript

a recorded run against the sample docs lives at [samples/demo_output.md](samples/demo_output.md). it shows multi-turn memory carrying context across follow-up questions, MMR-based retrieval pulling chunks from the right sources, and the guardrails blocking a prompt injection attempt at the end.

## stack

LangChain (text splitter), sentence-transformers (embeddings), FAISS (vector store), Anthropic SDK (LLM), Flask (HTTP), pytest. all Python.
