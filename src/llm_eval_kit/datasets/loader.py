"""
Dataset loaders — load evaluation samples from various sources.

load_jsonl: generic JSONL where each line maps to EvalSample fields.
load_bfcl: BFCL-specific JSONL with field mapping for the Berkeley
           Function Calling Leaderboard dataset.
"""
import json
import logging
from typing import Optional

from llm_eval_kit.models.messages import Message
from llm_eval_kit.models.datasets import EvalDataset, EvalSample

logger = logging.getLogger(__name__)


def load_jsonl(
    path: str, max_samples: Optional[int] = None
) -> EvalDataset:
    """Load a generic JSONL file into an EvalDataset."""
    return EvalDataset.from_jsonl(path, max_samples=max_samples)


def load_bfcl(
    path: str, max_samples: Optional[int] = None
) -> EvalDataset:
    """
    Load a BFCL JSONL file with field mapping.

    BFCL format (each line):
      - "id": unique identifier
      - "question": list of message dicts (the user prompt)
      - "function": list of tool definition dicts (JSON schemas)
      - ground truth: varies by file, often a separate answer file

    Note: BFCL files are NOT compatible with HuggingFace load_dataset.
    """
    samples = []
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

            if "id" not in data or "question" not in data:
                logger.warning(
                    "Skipping line %d: missing 'id' or 'question'", line_num
                )
                continue

            # Map BFCL fields to EvalSample
            question = data["question"]
            if isinstance(question, list):
                messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in question]
            else:
                # Some BFCL entries have question as a string
                messages = [Message(role="user", content=str(question))]

            sample = EvalSample(
                id=str(data["id"]),
                messages=messages,
                ground_truth=data.get("ground_truth"),
                metadata={
                    "tool_definitions": data.get("function", []),
                },
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

    return EvalDataset(samples)



def load_huggingface(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
    prompt_key: str = "prompt",
    response_key: str = "response",
    ground_truth_key: Optional[str] = "ground_truth",
    id_key: Optional[str] = "id",
    config_name: Optional[str] = None,
    data_files: Optional[str] = None,
) -> EvalDataset:
    """
    Load a dataset from HuggingFace Hub.

    Requires: pip install llm-eval-kit[datasets]

    Args:
        dataset_name: HF dataset name
            (e.g. "gorilla-llm/Berkeley-Function-Calling-Leaderboard")
        split: Dataset split (default: "train")
        max_samples: Max samples to load (None = all)
        token: HuggingFace API token. Falls back to HF_TOKEN env var.
        prompt_key: Column name containing the prompt/question
        response_key: Column name for model response (None to skip)
        ground_truth_key: Column name for ground truth (None to skip)
        id_key: Column name for sample ID (None to auto-generate)
        config_name: Dataset config/subset name (for multi-config
            datasets)
        data_files: Specific file(s) to load from the repo
            (e.g. "BFCL_v3_simple.json"). Useful when a HF repo
            contains multiple files with different schemas.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed. "
            "Run: uv pip install -e \".[datasets]\""
        )

    import os
    hf_token = token or os.environ.get("HF_TOKEN")

    # Build kwargs for load_dataset
    load_kwargs = {
        "split": split,
        "token": hf_token,
    }
    if config_name:
        load_kwargs["name"] = config_name
    if data_files:
        load_kwargs["data_files"] = data_files

    logger.info(
        "Loading %s (split=%s%s) from HuggingFace...",
        dataset_name,
        split,
        f", file={data_files}" if data_files else "",
    )
    ds = load_dataset(dataset_name, **load_kwargs)

    samples = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # Build sample ID
        sample_id = str(row.get(id_key, i)) if id_key else str(i)

        # Build messages from available columns
        messages = []
        if prompt_key and prompt_key in row:
            prompt = row[prompt_key]
            if isinstance(prompt, list):
                # Handle nested lists (e.g. BFCL "question"
                # is list[list[message_dict]])
                flat = prompt
                if (
                    flat
                    and isinstance(flat[0], list)
                ):
                    flat = flat[0]
                for msg in flat:
                    if isinstance(msg, dict):
                        messages.append(Message(**msg))
                    else:
                        messages.append(
                            Message(role="user", content=str(msg))
                        )
            else:
                messages.append(
                    Message(role="user", content=str(prompt))
                )

        if response_key and response_key in row:
            messages.append(
                Message(
                    role="assistant",
                    content=str(row[response_key]),
                )
            )

        # Ground truth
        gt = row.get(ground_truth_key) if ground_truth_key else None

        # Collect remaining columns as metadata
        skip_keys = {
            prompt_key, response_key, ground_truth_key, id_key,
        }
        metadata = {
            k: v for k, v in row.items() if k not in skip_keys
        }

        samples.append(EvalSample(
            id=sample_id,
            messages=messages,
            ground_truth=gt,
            metadata=metadata,
        ))

    logger.info(
        "Loaded %d samples from %s", len(samples), dataset_name,
    )
    return EvalDataset(samples)

