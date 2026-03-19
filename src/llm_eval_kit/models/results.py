"""
Evaluation result models — the output contract for all graders.

MetricResult and EvaluateResult are Pydantic v2 BaseModels (boundary models).
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """A single named metric score with explanation."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "reason": self.reason,
            "is_valid": self.is_valid,
        }


class EvaluateResult(BaseModel):
    """Complete output of a grader — overall score, sub-metrics, and metadata."""

    score: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None
    is_valid: bool = True
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Score: {self.score:.4f}"]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        for name, metric in self.metrics.items():
            lines.append(f"  {name}: {metric.score:.4f} ({metric.reason})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "reason": self.reason,
            "is_valid": self.is_valid,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def aggregate(cls, results: List["EvaluateResult"]) -> "EvaluateResult":
        """Compute mean score across a list of results."""
        if not results:
            return cls(score=0.0, reason="No results to aggregate")
        avg = sum(r.score for r in results) / len(results)
        return cls(
            score=avg,
            reason=f"Aggregated over {len(results)} samples",
        )
