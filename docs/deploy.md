# Lambda Deployment

Deploy any grader as an AWS Lambda function for use as a reward function in Bedrock RFT jobs. The deploy command packages your grader with all dependencies (including pydantic), creates or updates the Lambda, and wires up the handler automatically.

Requires `uv pip install -e ".[deploy]"`.

## 1. Create a Lambda Execution Role

If you don't already have one:

```bash
aws iam create-role \
  --role-name llm-eval-kit-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
```

## 2. Create a Config File

Create `llm_eval_kit.yaml` in your project root:

```yaml
aws:
  region: us-east-1
  account_id: "123456789012"
  lambda:
    function_name: my-reward-function
    runtime: python3.12
    timeout: 60
    memory_size: 256
    role_arn: arn:aws:iam::123456789012:role/llm-eval-kit-lambda-role
```

Config values can also be set via environment variables: `AWS_REGION`, `AWS_DEFAULT_REGION`, `AWS_ACCOUNT_ID`.

## 3. Deploy

```bash
# Deploy a built-in grader
llm-eval-kit deploy --grader exact_match

# Deploy a custom grader from a module path
llm-eval-kit deploy --grader-path my_module:my_grader

# Deploy with a specific config file
llm-eval-kit deploy --grader tool_call --config my_config.yaml
```

## 4. Test the Deployed Function

```bash
aws lambda invoke \
  --function-name my-reward-function \
  --payload '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ],
    "ground_truth": "4"
  }' \
  /dev/stdout
```

Expected response:

```json
{
  "statusCode": 200,
  "body": {
    "score": 1.0,
    "reason": "Exact match",
    "is_valid": true,
    "metrics": {"exact_match": {"score": 1.0, "reason": "case_sensitive=False", "is_valid": true}},
    "metadata": {}
  }
}
```

## What Happens Under the Hood

1. Your grader + the `llm_eval_kit` package + dependencies (pydantic, etc.) are installed for the Lambda runtime and zipped into a deployment package
2. An auto-generated `handler.py` wraps your grader with Lambda-compatible request/response handling
3. The Lambda function is created or updated in your AWS account
4. The function accepts `{"messages": [...], "ground_truth": ..., "kwargs": {}}` payloads

## Config Reference

### `llm_eval_kit.yaml`

```yaml
aws:
  region: us-east-1          # AWS region
  account_id: "123456789012" # AWS account ID (optional, for reference)
  lambda:
    function_name: my-reward-function  # Lambda function name
    runtime: python3.12                # Lambda runtime
    handler: handler.lambda_handler    # Handler path (default)
    timeout: 60                        # Timeout in seconds
    memory_size: 256                   # Memory in MB
    role_arn: arn:aws:iam::...         # Lambda execution role ARN
    environment: {}                    # Extra env vars for the Lambda
```

### Environment Variable Overrides

| Variable | Overrides |
|----------|-----------|
| `AWS_REGION` | `aws.region` |
| `AWS_DEFAULT_REGION` | `aws.region` |
| `AWS_ACCOUNT_ID` | `aws.account_id` |
