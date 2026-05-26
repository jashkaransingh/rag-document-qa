from .pipeline import retrieve
from .memory import get_session
from .guardrails import is_safe
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so explicitly — do not hallucinate.
Always cite which part of the context you're drawing from."""


def answer(query: str, session_id: str) -> dict:
    safe, reason = is_safe(query)
    if not safe:
        return {"answer": f"Query rejected: {reason}", "sources": []}

    memory = get_session(session_id)
    chunks = retrieve(query)

    if not chunks:
        return {"answer": "No documents have been indexed yet.", "sources": []}

    context = "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )

    history_str = memory.format_for_prompt()
    history_section = f"\n\nConversation history:\n{history_str}" if history_str else ""

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}
{history_section}

Question: {query}

Answer:"""

    response = model.generate_content(prompt)
    answer_text = response.text

    memory.add("user", query)
    memory.add("assistant", answer_text)

    return {
        "answer": answer_text,
        "sources": chunks[:3]
    }
