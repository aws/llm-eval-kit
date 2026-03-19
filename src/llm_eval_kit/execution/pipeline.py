"""
Evaluation pipeline — runs graders over datasets and collects results.

EvalPipeline is a plain class (composition: has a grader + dataset).
EvalReport is a @dataclass (internal summary container).
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List

from llm_eval_kit.graders.base import Grader
from llm_eval_kit.models.datasets import EvalDataset
from llm_eval_kit.models.results import EvaluateResult

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Summary of an evaluation run."""

    total_samples: int
    avg_score: float
    min_score: float
    max_score: float
    results: List[EvaluateResult] = field(repr=False)

    def to_jsonl(self, path: str) -> None:
        """Write each EvaluateResult as a JSON line."""
        with open(path, "w") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + "\n")

    def summary(self) -> str:
        return (
            f"Samples: {self.total_samples} | "
            f"Avg: {self.avg_score:.4f} | "
            f"Min: {self.min_score:.4f} | "
            f"Max: {self.max_score:.4f}"
        )


class EvalPipeline:
    """Orchestrates running a grader over a dataset."""

    def __init__(self, grader: Grader, dataset: EvalDataset) -> None:
        self.grader = grader
        self.dataset = dataset

    def run(self) -> List[EvaluateResult]:
        results = []
        for sample in self.dataset:
            try:
                result = self.grader.grade(
                    messages=sample.messages,
                    ground_truth=sample.ground_truth,
                    **sample.metadata,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Grader failed on sample %s: %s", sample.id, e)
                results.append(
                    EvaluateResult(
                        score=0.0,
                        is_valid=False,
                        reason=f"Error: {e}",
                    )
                )
        return results

    def run_with_report(self) -> EvalReport:
        results = self.run()
        scores = [r.score for r in results]
        return EvalReport(
            total_samples=len(results),
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            results=results,
        )
