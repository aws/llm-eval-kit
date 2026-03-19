"""Shared helpers for built-in graders."""
from typing import List, Optional

from llm_eval_kit.models.messages import Message


def get_last_assistant_content(messages: List[Message]) -> Optional[str]:
    """Walk messages in reverse, return first assistant content found."""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.content is not None:
            return msg.content
    return None
