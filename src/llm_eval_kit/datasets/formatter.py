"""
RFT dataset formatters — convert EvalDataset to Bedrock RFT training formats.

Two output formats supported:
  - Bedrock API: for create_model_customization_job (uploads to S3)
  - OpenAI-compatible: for client.files.create (uploads via API)

Both share the same core schema (messages + ground_truth) but differ
in metadata fields and how ground_truth is structured.
"""
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_eval_kit.models.datasets import EvalDataset, EvalSample

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Result of a train/val/test split with file paths."""

    train_path: str
    train_size: int
    val_path: Optional[str] = None
    val_size: int = 0
    test_path: Optional[str] = None
    test_size: int = 0
    paths: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"train={self.train_size}"]
        if self.val_size:
            parts.append(f"val={self.val_size}")
        if self.test_size:
            parts.append(f"test={self.test_size}")
        return " | ".join(parts)


def format_for_bedrock(
    sample: EvalSample,
    system_prompt: Optional[str] = None,
    domain: Optional[str] = None,
    data_source: Optional[str] = None,
    split_name: str = "train",
    index: int = 0,
) -> dict:
    """
    Format a single EvalSample for the Bedrock API RFT schema.

    Output schema:
        {
            "messages": [{"role": ..., "content": ...}, ...],
            "metadata": {"ground_truth": ...},
            "task_id": "...",
            "domain": "...",
            "data_source": "..."
        }
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in sample.messages:
        messages.append({"role": msg.role, "content": msg.content})

    row: Dict[str, Any] = {
        "messages": messages,
        "metadata": {
            "ground_truth": sample.ground_truth,
        },
    }

    # Optional metadata fields
    task_id = sample.id or f"{split_name}_{index}"
    row["task_id"] = task_id

    if domain:
        row["domain"] = domain
    if data_source:
        row["data_source"] = data_source

    # Pass through tool_definitions if present
    if "tool_definitions" in sample.metadata:
        row["metadata"]["tool_definitions"] = (
            sample.metadata["tool_definitions"]
        )

    return row


def format_for_openai(
    sample: EvalSample,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Format a single EvalSample for the OpenAI-compatible RFT schema.

    Output schema:
        {
            "messages": [{"role": ..., "content": ...}, ...],
            "ground_truth": "..."
        }

    The OpenAI-compatible path uses client.files.create() to upload,
    so no S3 or task_id/domain fields are needed.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in sample.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Ground truth — flatten to string if it's a list with one item
    gt = sample.ground_truth
    if isinstance(gt, list) and len(gt) == 1:
        gt = str(gt[0])
    elif isinstance(gt, list):
        gt = json.dumps(gt)

    return {
        "messages": messages,
        "ground_truth": gt,
    }


def export_rft_jsonl(
    dataset: EvalDataset,
    output_dir: str,
    fmt: str = "bedrock",
    system_prompt: Optional[str] = None,
    domain: Optional[str] = None,
    data_source: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> SplitResult:
    """
    Export an EvalDataset to RFT-formatted JSONL files with
    train/val/test split.

    Args:
        dataset: The EvalDataset to export.
        output_dir: Directory to write JSONL files into.
        fmt: "bedrock" for Bedrock API or "openai" for
            OpenAI-compatible API.
        system_prompt: Optional system message prepended to each
            sample's messages.
        domain: Domain tag (Bedrock format only).
        data_source: Data source tag (Bedrock format only).
        train_ratio: Fraction of data for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
            Remainder goes to test.
        seed: Random seed for shuffling.
        shuffle: Whether to shuffle before splitting.

    Returns:
        SplitResult with file paths and counts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = list(dataset)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    total = len(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    splits: List[Tuple[str, List[EvalSample]]] = [
        ("train", samples[:train_size]),
    ]
    if val_size > 0:
        splits.append(
            ("val", samples[train_size:train_size + val_size])
        )
    if test_size > 0:
        splits.append(("test", samples[train_size + val_size:]))

    paths: Dict[str, str] = {}
    for split_name, split_samples in splits:
        path = out / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(split_samples):
                if fmt == "openai":
                    row = format_for_openai(
                        sample,
                        system_prompt=system_prompt,
                    )
                else:
                    row = format_for_bedrock(
                        sample,
                        system_prompt=system_prompt,
                        domain=domain,
                        data_source=data_source,
                        split_name=split_name,
                        index=i,
                    )
                f.write(json.dumps(row) + "\n")
        paths[split_name] = str(path)
        logger.info(
            "Wrote %d samples to %s", len(split_samples), path,
        )

    return SplitResult(
        train_path=paths["train"],
        train_size=train_size,
        val_path=paths.get("val"),
        val_size=val_size,
        test_path=paths.get("test"),
        test_size=test_size,
        paths=paths,
    )


def upload_to_s3(
    split_result: SplitResult,
    bucket: str,
    prefix: str,
    session=None,
) -> Dict[str, str]:
    """
    Upload split JSONL files to S3.

    Args:
        split_result: Output from export_rft_jsonl.
        bucket: S3 bucket name.
        prefix: S3 key prefix (e.g. "rft-data/bfcl").
        session: Optional boto3.Session. Uses default if None.

    Returns:
        Dict mapping split name to S3 URI.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 not installed. "
            "Run: uv pip install -e \".[deploy]\""
        )

    if session is None:
        session = boto3.Session()

    s3 = session.client("s3")
    uris: Dict[str, str] = {}

    for split_name, local_path in split_result.paths.items():
        key = f"{prefix.rstrip('/')}/{split_name}.jsonl"
        s3.upload_file(local_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        uris[split_name] = uri
        logger.info("Uploaded %s → %s", local_path, uri)

    return uris
