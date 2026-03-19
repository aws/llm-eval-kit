# Graders

Graders are the core evaluation unit in llm-eval-kit. A grader takes a conversation (messages) and ground truth, then returns a scored `EvaluateResult`.

## Built-in Graders

| Name | Description |
|------|-------------|
| `exact_match` | Exact string comparison (case-insensitive by default) |
| `string_similarity` | Levenshtein distance or token F1 fuzzy matching |
| `tool_call` | BFCL-style AST comparison of function calls with type coercion |

List them from the CLI:

```bash
llm-eval-kit list-graders
```

## Writing a Custom Grader

Use the `@grader` decorator to register a function as a grader:

```python
from llm_eval_kit.graders.decorator import grader
from llm_eval_kit.models.results import EvaluateResult

@grader(name="my_grader", description="My custom grader")
def my_grader(messages, ground_truth, **kwargs):
    response = messages[-1].content
    match = response.strip().lower() == str(ground_truth).strip().lower()
    return EvaluateResult(
        score=1.0 if match else 0.0,
        reason="Match" if match else "No match",
    )
```

Your function receives:
- `messages` — list of `Message` objects (role + content)
- `ground_truth` — the expected answer (str, list, or dict)
- `**kwargs` — any extra metadata from the sample

It must return an `EvaluateResult` with at minimum a `score` (0.0–1.0).

## Using a Custom Grader from CLI

Point to your grader with `--grader-path`:

```bash
llm-eval-kit evaluate \
    --grader-path my_module:my_grader \
    --data samples.jsonl
```

The format is `module.path:function_name`. The module must be importable from your current directory or installed in your environment.

## Grader Architecture

- `Grader` (ABC) — base class with a `grade(messages, ground_truth, **kwargs)` method
- `@grader` decorator — wraps a plain function into a `_FunctionGrader` instance
- `GraderRegistry` — singleton that maps names to grader instances
- Built-in graders auto-register on import via `graders/__init__.py`

## EvaluateResult

```python
from llm_eval_kit.models.results import EvaluateResult, MetricResult

result = EvaluateResult(
    score=0.85,
    reason="Partial match",
    is_valid=True,
    metrics={
        "name_accuracy": MetricResult(score=1.0, reason="Correct name"),
        "value_accuracy": MetricResult(score=0.7, reason="2/3 values matched"),
    },
    metadata={"debug": "extra info"},
)
```
