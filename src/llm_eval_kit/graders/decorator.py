"""
The @grader decorator — wraps plain functions into Grader instances.

Supports both @grader and @grader(name=..., description=...) syntax.
"""
import functools
import inspect
from typing import Any, Callable, List, Optional, Union

from llm_eval_kit.models.messages import Message
from llm_eval_kit.models.results import EvaluateResult
from .base import Grader


class _FunctionGrader(Grader):
    """Internal: wraps a plain function as a Grader. Created by @grader."""

    def __init__(
        self, fn: Callable, grader_name: str, grader_desc: str
    ) -> None:
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._name = grader_name
        self._description = grader_desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def grade(
        self,
        messages: List[Message],
        ground_truth: Optional[Union[str, dict, list]] = None,
        **kwargs: Any,
    ) -> EvaluateResult:
        return self._fn(messages, ground_truth, **kwargs)


def _validate_signature(fn: Callable) -> None:
    """Check that fn accepts (messages, ground_truth, **kwargs)."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    if len(params) < 2:
        raise TypeError(
            f"Grader function '{fn.__name__}' must accept at least "
            f"(messages, ground_truth, **kwargs), got: ({', '.join(params)})"
        )
    if params[0] != "messages" or params[1] != "ground_truth":
        raise TypeError(
            f"Grader function '{fn.__name__}' first two parameters must be "
            f"'messages' and 'ground_truth', got: ({', '.join(params[:2])})"
        )


def grader(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[Grader, Callable[..., Grader]]:
    """
    Decorator to turn a function into a Grader.

    Usage:
        @grader
        def my_grader(messages, ground_truth, **kwargs): ...

        @grader(name="custom", description="My grader")
        def my_grader(messages, ground_truth, **kwargs): ...
    """
    def _wrap(fn: Callable) -> _FunctionGrader:
        _validate_signature(fn)
        return _FunctionGrader(
            fn,
            grader_name=name or fn.__name__,
            grader_desc=description or fn.__doc__ or "",
        )

    if func is not None:
        # Bare @grader (no parentheses)
        return _wrap(func)
    # @grader(...) with arguments — return the wrapper
    return _wrap
