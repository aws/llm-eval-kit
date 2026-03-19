"""
llm-eval-kit — A Python SDK for LLM evaluation and RFT grader development.

This package provides:
- Grader framework: Define evaluation functions with @grader decorator
- Built-in graders: exact_match, string_similarity, llm_judge
- Dataset loading: JSONL and HuggingFace dataset support
- Evaluation pipeline: Run graders over datasets and collect results
- SageMaker integration: Pre/post processing for SageMaker eval jobs (Lambda)
- CLI: Command-line tools for running evaluations

Quick start:
    from llm_eval_kit.graders.decorator import grader
    from llm_eval_kit.models.results import EvaluateResult

    @grader
    def my_grader(messages, ground_truth, **kwargs):
        # Your evaluation logic here
        return EvaluateResult(score=1.0, reason="Perfect!")
"""

__version__ = "1.1.0"
