# llm-eval-kit — RLVR Grader Framework

A grader framework for building evaluation functions, running them over datasets, and deploying them as AWS Lambda reward functions for RLVR workflows on Amazon Bedrock.

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[datasets]"  # HuggingFace dataset support
uv pip install -e ".[deploy]"    # AWS Lambda deployment
```

## Quick Start

Write a grader:

```python
from llm_eval_kit.graders.decorator import grader
from llm_eval_kit.models.results import EvaluateResult

@grader
def my_grader(messages, ground_truth, **kwargs):
    response = messages[-1].content
    match = response.strip().lower() == str(ground_truth).strip().lower()
    return EvaluateResult(
        score=1.0 if match else 0.0,
        reason="Match" if match else "No match",
    )
```

Run it:

```bash
llm-eval-kit evaluate --grader exact_match --data samples.jsonl
```

Or from Python:

```python
from llm_eval_kit.datasets.loader import load_jsonl
from llm_eval_kit.execution.pipeline import EvalPipeline
from llm_eval_kit.graders import exact_match_grader

dataset = load_jsonl("samples.jsonl")
report = EvalPipeline(exact_match_grader, dataset).run_with_report()
print(report.summary())
# Samples: 2 | Avg: 1.0000 | Min: 1.0000 | Max: 1.0000
```

## Documentation

| Topic | Description |
|-------|-------------|
| [Graders](../../docs/graders.md) | Built-in graders, writing custom graders, the `@grader` decorator |
| [Datasets](../../docs/datasets.md) | Loading from JSONL, BFCL, and HuggingFace Hub |
| [Lambda Deployment](../../docs/deploy.md) | Deploy graders as AWS Lambda reward functions for RLVR |
| [CLI Reference](../../docs/cli.md) | All CLI commands and options |

## Built-in Graders

| Name | Description |
|------|-------------|
| `exact_match` | Exact string comparison (case-insensitive by default) |
| `string_similarity` | Levenshtein distance or token F1 fuzzy matching |
| `tool_call` | BFCL-style AST comparison of function calls with type coercion |

## Project Structure

```
llm_eval_kit/
├── models/           # Pydantic data models (Message, EvaluateResult, EvalSample)
├── graders/          # Grader framework (ABC, decorator, registry)
│   └── builtins/     # Built-in grader implementations
├── datasets/         # Dataset loaders (JSONL, BFCL, HuggingFace)
├── execution/        # Evaluation pipeline and reporting
├── deploy/           # AWS Lambda deployment
├── cli/              # Command-line interface
├── utils/            # Dynamic module loading
├── processors/       # SageMaker pre/post processing (existing)
├── model/            # SageMaker payload models (existing)
└── lambda_handler.py # SageMaker Lambda handler (existing)
```

## License

Apache-2.0
