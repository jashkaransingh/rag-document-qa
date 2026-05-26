from collections import deque
from typing import List, Dict


class ConversationMemory:
    """
    Sliding window conversation memory.
    Keeps the last N turns to inject into the system prompt.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict]:
        return list(self.history)

    def format_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()


# In-memory store keyed by session_id
_sessions: Dict[str, ConversationMemory] = {}


def get_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def delete_session(session_id: str):
    _sessions.pop(session_id, None)
