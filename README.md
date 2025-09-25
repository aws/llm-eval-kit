# Nova Custom Evaluation SDK

A Python SDK for creating custom evaluation metrics for AWS Nova on-demand model evaluation on Sagemaker Training Job with built-in Pydantic validation.

## Installation

```
git clone https://github.com/aws/nova-custom-eval-sdk.git
cd nova-custom-eval-sdk
pip install .
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
You need to create a lambda (follow this [guide](https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html)) and upload `nova-custom-eval-sdk` as a lambda layer in order to use it.

In the github release, you should be able to find a pre-built nov-custom-eval-layer.zip file.

Use below command to upload custom lambda layer.

```
aws lambda publish-layer-version \
    --layer-name nova-custom-sdk-layer \
    --zip-file fileb://nova-layer.zip \
    --compatible-runtimes python3.12 python3.11 python3.10 python3.9
```
You need to add this layer as custom layer along with an AWS layer: `AWSLambdaPowertoolsPythonV3-python312-arm64` (because of pydantic depencency) to your lambda.

Then update your lambda code with:

```python
from nova_custom_evaluation_sdk.processors.decorators import preprocess, postprocess
from nova_custom_evaluation_sdk.lambda_handler import build_lambda_handler

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
def postprocessor(event: list, context) -> dict:
    # data is already validated and extracted from event
    data = event.get('data', [])
    inference_output = data.get('inference_output', '')
    gold = data.get('gold', '')
    
    metrics = []
    accuracy = 1.0 if inference_output.lower() == gold.lower() else 0.0
    metrics.append({
        "metric": "inverted_accuracy_custom",
        "value": accuracy
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
    "system": "You are a helpful assistant" // optional
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

## Testing

```bash
# Run all tests
python -m pytest -v

# Run example
python example/run_example.py
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests with coverage
python -m pytest tests/ --cov=nova_custom_evaluation_sdk
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

