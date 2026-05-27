"""
Multi-turn conversation memory.

The default is a sliding window of the last N turns. The window gets injected
into the system prompt before each new query, so follow-up questions can
reference earlier exchanges without re-stating context.

This is intentionally simple. More elaborate memory schemes (summarization,
vector-indexed history, entity tracking) exist and are sometimes worth it, but
for typical document QA over a few turns the sliding window is the right
tradeoff between answer quality and complexity.

Sessions are keyed by an opaque session_id supplied by the caller.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, window_turns: int = 6):
        """window_turns counts user+assistant pairs to keep in memory."""
        self.window_turns = window_turns
        self._sessions: Dict[str, Deque[Turn]] = defaultdict(
            lambda: deque(maxlen=window_turns * 2))

    def add_user(self, session_id: str, content: str) -> None:
        self._sessions[session_id].append(Turn("user", content))

    def add_assistant(self, session_id: str, content: str) -> None:
        self._sessions[session_id].append(Turn("assistant", content))

    def get_history(self, session_id: str) -> List[Turn]:
        return list(self._sessions[session_id])

    def format_for_prompt(self, session_id: str) -> str:
        """Render the conversation history as a plain text block for prompting."""
        turns = self.get_history(session_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            label = "User" if t.role == "user" else "Assistant"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def session_count(self) -> int:
        return len(self._sessions)
