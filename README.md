# LLM Eval Kit

A Python SDK for creating custom evaluation metrics for LLM model evaluation on Sagemaker Training Job with built-in Pydantic validation.

For the official integration with AWS Sagemaker training job, please view in the [Official AWS Sagemaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html).

## Installation

```bash
git clone https://github.com/aws/llm-eval-kit.git
cd llm-eval-kit
uv venv .venv && source .venv/bin/activate
uv pip install .
```

## Architecture

The SDK provides:

- **Pydantic Validation**: Automatic input/output validation using Pydantic models
- **PreProcessor**: For input data transformation with validation
- **PostProcessor**: For output data formatting with validation
- **Decorators**: Simplified processor creation (@preprocess, @postprocess)
- **Lambda Handler Builder**: Easy Lambda function creation
- **Exception Handling**: Custom error types with validation feedback

## Quick Start

### Complete Example

See `example/run_example.py` for a complete working example to run locally.

### Run in AWS Lambda

You need to create a lambda (follow this [guide](https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html)) and upload `llm-eval-kit` as a lambda layer in order to use it.

In the [github release](https://github.com/aws/llm-eval-kit/releases), you should be able to find a pre-built llm-eval-kit-layer.zip file.

Use below command to upload custom lambda layer.

```
aws lambda publish-layer-version \
--layer-name llm-eval-kit-layer \
--zip-file fileb://llm-eval-kit-layer.zip \
--compatible-runtimes python3.12 python3.11 python3.10 python3.9
```

You need to add this layer as custom layer along with the required AWS layer: `AWSLambdaPowertoolsPythonV3-python312-arm64` (because of pydantic depencency) to your lambda.

Then update your lambda code with:

```python
from llm_eval_kit.processors.decorators import preprocess, postprocess
from llm_eval_kit.lambda_handler import build_lambda_handler

@preprocess
def preprocessor(event: dict, context) -> dict:
    data = event.get('data', {})
    return {
        "statusCode": 200,
        "body": {
            "system": data.get("system"),
            "prompt": data.get("prompt", ""),
            "gold": data.get("gold", "")
        }
    }

@postprocess
def postprocessor(event: dict, context) -> dict:
    # data is already validated and extracted from event
    data = event.get('data', [])
    inference_output = data.get('inference_output', '')
    gold = data.get('gold', '')

    metrics = []
    inverted_accuracy = 0 if inference_output.lower() == gold.lower() else 1.0
    metrics.append({
        "metric": "inverted_accuracy_custom",
        "value": inverted_accuracy
    })
    # Add more metrics here

    return {
        "statusCode": 200,
        "body": metrics
    }

# Build Lambda handler
lambda_handler = build_lambda_handler(
    preprocessor=preprocessor,
    postprocessor=postprocessor
)
```

## Input/Output Validation

The SDK automatically validates:

### Preprocessing Input

```json
{
    "process_type": "preprocess",
    "data": {
        "prompt": "what can you do?",
        "gold": "Hello! How can I help you today?",
        "system": "You are a helpful assistant"
    }
}
```

### Postprocessing Input

```json
{
    "process_type": "postprocess",
    "data": [
        {
            "prompt": "what can you do",
            "inference_output": "Hello! How can I help you today?",
            "gold": "Hello! How can I help you today?"
        }
    ]
}
```

## RLVR Grader Framework

llm-eval-kit also includes a grader framework for building and deploying reward functions for Reinforcement Learning with Verifiable Rewards (RLVR) on Amazon Bedrock. This extends the SDK beyond SageMaker evaluation into RFT (Reinforcement Fine-Tuning) workflows.

Features include:

- Built-in graders for exact match, string similarity, and BFCL tool-calling evaluation
- A `@grader` decorator for writing custom reward functions
- Dataset loaders for JSONL, BFCL, and HuggingFace Hub
- One-command Lambda deployment of graders as reward functions
- A CLI for local evaluation, validation, and deployment

```bash
uv pip install -e ".[dev,datasets,deploy]"
```

For full documentation on the grader framework, see the [src/llm_eval_kit README](src/llm_eval_kit/README.md).

| Topic | Description |
|-------|-------------|
| [Graders](docs/graders.md) | Built-in graders, writing custom graders, the `@grader` decorator |
| [Datasets](docs/datasets.md) | Loading from JSONL, BFCL, and HuggingFace Hub |
| [Lambda Deployment](docs/deploy.md) | Deploy graders as AWS Lambda reward functions for RLVR |
| [CLI Reference](docs/cli.md) | All CLI commands and options |

## Testing

```bash
# Run all tests
python -m pytest -v

# Run example
python example/run_example.py
```

## Development

```bash
# Create venv and install in development mode
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,datasets,deploy]"

# Run tests with coverage
python -m pytest tests/ --cov=llm_eval_kit
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
