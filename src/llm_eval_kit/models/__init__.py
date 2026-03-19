"""
Core data models for llm-eval-kit.

Pydantic v2 BaseModel: Message, MetricResult, EvaluateResult, EvalSample
Plain class: Conversation, EvalDataset
"""
from .messages import Conversation, Message
from .results import EvaluateResult, MetricResult
from .datasets import EvalDataset, EvalSample

__all__ = [
    "Message",
    "Conversation",
    "MetricResult",
    "EvaluateResult",
    "EvalSample",
    "EvalDataset",
]
