"""
Grader framework — the core evaluation engine of llm-eval-kit.

Importing this package auto-populates the default_registry with built-in graders.
"""
from .base import Grader
from .decorator import grader
from .registry import GraderRegistry, default_registry
from .builtins.exact_match import exact_match_grader
from .builtins.string_similarity import string_similarity_grader
from .builtins.tool_call import tool_call_grader

# Auto-register built-in graders
default_registry.register("exact_match", exact_match_grader)
default_registry.register("string_similarity", string_similarity_grader)
default_registry.register("tool_call", tool_call_grader)

__all__ = [
    "Grader",
    "grader",
    "GraderRegistry",
    "default_registry",
    "exact_match_grader",
    "string_similarity_grader",
    "tool_call_grader",
]
