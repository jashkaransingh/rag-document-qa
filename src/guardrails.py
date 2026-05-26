import re

# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore (previous|all|prior) instructions",
    r"disregard (the|your) system prompt",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as (if|a|an)",
    r"forget (everything|all)",
    r"new instructions:",
    r"\\n\\nHuman:",
    r"<\|system\|>",
]

MAX_QUERY_LENGTH = 2000


def is_safe(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Checks for prompt injection and length limits.
    """
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})"

    lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return False, f"Potential prompt injection detected"

    return True, ""
