from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ChatHistory:
    """
    Lightweight conversational memory for multi-turn RAG.

    Stores recent question/answer pairs and formats them for inclusion
    in the LLM prompt so follow-up questions ("tell me more", "what about
    the exceptions?") have the necessary context.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    max_turns: int = 5

    def add_turn(self, question: str, answer: str):
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2) :]

    def to_context(self) -> str:
        if not self.messages:
            return ""
        lines = ["CONVERSATION HISTORY:"]
        for msg in self.messages:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            lines.append(f"{prefix}{msg['content'][:300]}")
        return "\n".join(lines) + "\n"
