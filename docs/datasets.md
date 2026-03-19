# Datasets

llm-eval-kit supports loading evaluation data from JSONL files, BFCL-formatted files, and HuggingFace Hub.

## JSONL Format

Each line is a JSON object with `id`, `messages`, and `ground_truth`:

```jsonl
{"id": "1", "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}], "ground_truth": "4"}
{"id": "2", "messages": [{"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris"}], "ground_truth": "Paris"}
```

Load from CLI:

```bash
llm-eval-kit evaluate --grader exact_match --data samples.jsonl
```

Load from Python:

```python
from llm_eval_kit.datasets.loader import load_jsonl

dataset = load_jsonl("samples.jsonl", max_samples=100)
```

Validate a file before running:

```bash
llm-eval-kit validate --data samples.jsonl
```

## BFCL Format

The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) uses a specific JSONL format with `id`, `question` (list of message dicts), and `function` (tool definitions).

```bash
llm-eval-kit evaluate \
    --grader tool_call \
    --data BFCL_v3_multiple.json \
    --format bfcl
```

```python
from llm_eval_kit.datasets.loader import load_bfcl

dataset = load_bfcl("BFCL_v3_multiple.json", max_samples=100)
```

## HuggingFace Hub

Pull datasets directly from HuggingFace. Requires `uv pip install -e ".[datasets]"`.

```python
from llm_eval_kit.datasets.loader import load_huggingface

dataset = load_huggingface(
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    split="train",
    max_samples=50,
    data_files="BFCL_v3_exec_simple.json",  # pick a specific file
    prompt_key="question",
    ground_truth_key="ground_truth",
    id_key="id",
    response_key=None,
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | (required) | HF dataset name (e.g. `"gorilla-llm/Berkeley-Function-Calling-Leaderboard"`) |
| `split` | `"train"` | Dataset split |
| `max_samples` | `None` | Limit number of samples |
| `token` | `None` | HF API token (falls back to `HF_TOKEN` env var) |
| `data_files` | `None` | Specific file(s) to load from the repo |
| `config_name` | `None` | Dataset config/subset name |
| `prompt_key` | `"prompt"` | Column name for the prompt |
| `response_key` | `"response"` | Column name for model response (`None` to skip) |
| `ground_truth_key` | `"ground_truth"` | Column name for ground truth (`None` to skip) |
| `id_key` | `"id"` | Column name for sample ID (`None` to auto-generate) |

### BFCL on HuggingFace

The BFCL repo has ~49 files with different schemas. You must use `data_files` to select one — loading the entire repo will fail.

Available files include: `BFCL_v3_simple.json`, `BFCL_v3_multiple.json`, `BFCL_v3_parallel.json`, `BFCL_v3_exec_simple.json`, `BFCL_v3_live_simple.json`, and more.

### Private/Gated Datasets

```python
dataset = load_huggingface(
    "my-org/my-private-dataset",
    token="hf_...",  # or set HF_TOKEN env var
)
```
