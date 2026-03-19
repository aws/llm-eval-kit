"""
Message and Conversation models for representing LLM interactions.

Message is a Pydantic v2 BaseModel (boundary model — validated, serializable).
Conversation is a plain Python class (lightweight wrapper — no validation overhead).
"""
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, model_validator


class Message(BaseModel):
    """Provider-agnostic chat message model."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None

    @model_validator(mode="after")
    def validate_tool_message(self) -> "Message":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "Messages with role 'tool' must include tool_call_id"
            )
        return self

    def to_openai_format(self) -> dict:
        """Return dict with only non-None fields, compatible with OpenAI API."""
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def from_openai(cls, data: dict) -> "Message":
        """Construct a Message from an OpenAI-format dict."""
        return cls(**data)


class Conversation:
    """Lightweight wrapper around a list of Messages with helper accessors."""

    def __init__(self, messages: List[Message]) -> None:
        self.messages = list(messages)

    def get_last_assistant_message(self) -> Optional[Message]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def get_system_prompt(self) -> Optional[str]:
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return None

    def to_openai_format(self) -> List[dict]:
        return [msg.to_openai_format() for msg in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
