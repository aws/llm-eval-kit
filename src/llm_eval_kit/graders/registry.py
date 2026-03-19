"""Grader registry — central place to register and discover graders by name."""
from typing import Dict, List

from .base import Grader


class GraderRegistry:
    """Maps string names to Grader instances."""

    def __init__(self) -> None:
        self._graders: Dict[str, Grader] = {}

    def register(self, name: str, grader: Grader) -> None:
        self._graders[name] = grader

    def get(self, name: str) -> Grader:
        if name not in self._graders:
            available = ", ".join(sorted(self._graders.keys()))
            raise KeyError(
                f"Grader '{name}' not found. Available: {available}"
            )
        return self._graders[name]

    def list_graders(self) -> List[str]:
        return sorted(self._graders.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._graders


default_registry = GraderRegistry()
