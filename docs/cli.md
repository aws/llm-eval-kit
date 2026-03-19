# CLI Reference

```
llm-eval-kit <command> [options]
```

## `evaluate`

Run a grader over a dataset.

```bash
llm-eval-kit evaluate --grader <name> --data <path> [options]
```

| Option | Description |
|--------|-------------|
| `--grader` | Built-in grader name (`exact_match`, `string_similarity`, `tool_call`) |
| `--grader-path` | Custom grader as `module.path:function_name` |
| `--data` | Path to JSONL dataset file (required) |
| `--format` | `jsonl` (default) or `bfcl` for BFCL-formatted files |
| `--output` | Write per-sample results to a JSONL file |
| `--max-samples` | Limit number of samples to evaluate |

Examples:

```bash
# Built-in grader
llm-eval-kit evaluate --grader exact_match --data samples.jsonl

# Custom grader with output
llm-eval-kit evaluate --grader-path my_module:my_grader --data samples.jsonl --output results.jsonl

# BFCL format with sample limit
llm-eval-kit evaluate --grader tool_call --data BFCL_v3_simple.json --format bfcl --max-samples 50
```

## `list-graders`

Show all registered graders.

```bash
llm-eval-kit list-graders
```

## `validate`

Check a dataset file for schema errors.

```bash
llm-eval-kit validate --data <path>
```

## `deploy`

Deploy a grader as an AWS Lambda function. Requires `uv pip install -e ".[deploy]"`.

```bash
llm-eval-kit deploy --grader <name> [options]
```

| Option | Description |
|--------|-------------|
| `--grader` | Built-in grader name |
| `--grader-path` | Custom grader as `module.path:function_name` |
| `--config` | Path to `llm_eval_kit.yaml` config file |

See [deploy.md](deploy.md) for the full deployment walkthrough.
