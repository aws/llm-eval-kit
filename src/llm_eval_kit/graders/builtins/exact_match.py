"""Exact match grader — checks if assistant response exactly matches ground truth."""
from llm_eval_kit.graders.decorator import grader
from llm_eval_kit.models.results import EvaluateResult, MetricResult
from ._helpers import get_last_assistant_content


@grader(
    name="exact_match",
    description="Exact string match between response and ground truth",
)
def exact_match_grader(messages, ground_truth, *, case_sensitive=False, **kwargs):
    response = get_last_assistant_content(messages)
    if response is None:
        return EvaluateResult(
            score=0.0,
            is_valid=False,
            reason="No assistant message found",
        )

    response = response.strip()
    expected = str(ground_truth).strip() if ground_truth is not None else ""

    if not case_sensitive:
        match = response.lower() == expected.lower()
    else:
        match = response == expected

    score = 1.0 if match else 0.0
    return EvaluateResult(
        score=score,
        reason="Exact match" if match else "No match",
        metrics={
            "exact_match": MetricResult(
                score=score,
                reason=f"case_sensitive={case_sensitive}",
            ),
        },
    )
