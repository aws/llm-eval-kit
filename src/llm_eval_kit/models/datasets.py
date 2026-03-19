"""
Dataset sample models.

EvalSample is a Pydantic v2 BaseModel (boundary — validated from external data).
EvalDataset is a plain Python class (internal container — lightweight).
"""
import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from .messages import Message

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """One evaluation sample with messages, ground truth, and metadata."""

    id: str
    messages: List[Message]
    ground_truth: Optional[Union[str, dict, list]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalDataset:
    """Container wrapping a list of EvalSample with Pythonic iteration."""

    def __init__(self, samples: List[EvalSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        max_samples: Optional[int] = None,
    ) -> "EvalDataset":
        """Read a JSONL file into EvalSample objects. Skips malformed lines."""
        samples: List[EvalSample] = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", line_num)
                    continue
                try:
                    samples.append(EvalSample(**data))
                except Exception as e:
                    logger.warning("Skipping invalid sample at line %d: %s", line_num, e)
                    continue
                if max_samples and len(samples) >= max_samples:
                    break
        return cls(samples)
