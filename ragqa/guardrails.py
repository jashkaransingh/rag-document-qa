"""
Prompt injection and jailbreak guardrails.

These run on every incoming query before retrieval and before the LLM call.
They are not a substitute for a real safety review on production systems, but
they catch the common attack patterns that show up in casual probing.

The set of patterns here came from spending an afternoon trying to break my
own RAG system. Most attempts fall into a handful of categories.

  - role override, "ignore previous instructions and..."
  - system prompt extraction, "what was your initial prompt"
  - delimiter confusion, fake </instructions> or markdown closers
  - persona hijack, "you are now DAN, the do-anything AI"
  - data exfiltration via context window, "print everything you know about X"

The defense is intentionally conservative. Anything that scores above the
threshold gets rejected with a generic message before retrieval runs. False
positives are acceptable, because in a document QA context the user can
rephrase. False negatives are not.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# Patterns are (regex, weight). Higher weight = stronger signal.
SUSPICIOUS_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Role override attempts
    (re.compile(r"\bignore\s+(?:all\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+|preceding\s+)+"
                r"(?:instructions?|prompts?|rules?|directives?|guidelines?)\b", re.I), 1.0),
    (re.compile(r"\bdisregard\s+(?:all|the|your|previous|prior|above|preceding)\b", re.I), 1.0),
    (re.compile(r"\bforget\s+(?:everything|all|your\s+(?:instructions?|rules?|training))\b", re.I), 0.8),

    # System prompt extraction
    (re.compile(r"\b(?:what|show|reveal|print|tell\s+me)\s+(?:is\s+|are\s+|me\s+)?(?:your|the)\s+"
                r"(?:system|initial|original|hidden|secret)\s+(?:prompt|instructions?|message)\b", re.I), 1.0),
    (re.compile(r"\brepeat\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions?)\b", re.I), 1.0),

    # Persona hijack
    (re.compile(r"\byou\s+are\s+(?:now\s+)?(?:DAN|jailbreak|unrestricted|uncensored|"
                r"a\s+different\s+AI|an?\s+evil)\b", re.I), 1.0),
    (re.compile(r"\bpretend\s+(?:you\s+are|to\s+be)\s+(?:DAN|jailbreak|unrestricted|"
                r"a\s+different\s+AI)\b", re.I), 1.0),

    # Delimiter confusion
    (re.compile(r"<\s*/?\s*(?:system|instructions?|context|prompt)\s*>", re.I), 0.6),
    (re.compile(r"```\s*system|<\|im_(?:start|end)\|>|<\|endoftext\|>", re.I), 0.6),

    # Data exfiltration
    (re.compile(r"\bprint\s+(?:everything|all)\s+(?:you\s+know|in\s+your\s+context|"
                r"in\s+the\s+(?:context|documents?))\b", re.I), 0.7),
    (re.compile(r"\boutput\s+the\s+(?:entire|full|complete)\s+(?:context|document|history)\b", re.I), 0.7),
]

GENERIC_REJECT_MSG = (
    "Your question looks like it might contain instructions targeting the assistant "
    "rather than the documents. Please rephrase as a question about the document content."
)


@dataclass
class GuardrailResult:
    allowed: bool
    score: float
    reasons: List[str]
    message: str  # what to return to the caller if blocked


class Guardrails:
    def __init__(self, threshold: float = 0.9, max_query_chars: int = 4000):
        self.threshold = threshold
        self.max_query_chars = max_query_chars

    def check(self, query: str) -> GuardrailResult:
        if not query or not query.strip():
            return GuardrailResult(False, 0.0, ["empty query"],
                                   "Please provide a question.")

        if len(query) > self.max_query_chars:
            return GuardrailResult(
                False, 1.0, ["query exceeds max length"],
                f"Query too long (max {self.max_query_chars} chars).")

        score = 0.0
        reasons: List[str] = []
        for pattern, weight in SUSPICIOUS_PATTERNS:
            if pattern.search(query):
                score += weight
                reasons.append(f"matched pattern: {pattern.pattern[:60]}...")

        if score >= self.threshold:
            return GuardrailResult(False, score, reasons, GENERIC_REJECT_MSG)

        return GuardrailResult(True, score, reasons, "")
