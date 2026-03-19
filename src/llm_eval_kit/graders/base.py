"""Abstract base class for all graders."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from llm_eval_kit.models.messages import Message
from llm_eval_kit.models.results import EvaluateResult


class Grader(ABC):
    """
    Interface that all graders must implement.

    Subclasses provide name, description, and grade().
    Every Grader is callable — __call__ delegates to grade().
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def grade(
        self,
        messages: List[Message],
        ground_truth: Optional[Union[str, dict, list]] = None,
        **kwargs: Any,
    ) -> EvaluateResult: ...

    def __call__(
        self,
        messages: List[Message],
        ground_truth: Optional[Union[str, dict, list]] = None,
        **kwargs: Any,
    ) -> EvaluateResult:
        return self.grade(messages, ground_truth, **kwargs)
