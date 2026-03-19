"""
AWS Lambda deployment for grader functions.

Packages a grader as a Lambda function and deploys it using boto3.
The deployed Lambda accepts a JSON payload with messages and ground_truth,
runs the grader, and returns the EvaluateResult.
"""
import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

from .config import DeployConfig, load_deploy_config

logger = logging.getLogger(__name__)

# Template for the Lambda handler that wraps a grader
HANDLER_TEMPLATE = '''"""Auto-generated Lambda handler for llm-eval-kit grader."""
import json
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from llm_eval_kit.models.messages import Message
from llm_eval_kit.models.results import EvaluateResult
from llm_eval_kit.utils.module_loader import load_function


# Load the grader at cold start
_GRADER_REF = os.environ.get("GRADER_REF", "{grader_ref}")
_grader = load_function(_GRADER_REF)


def lambda_handler(event, context):
    """
    Lambda handler for reward function evaluation.

    Expected payload:
    {{
        "messages": [{{"role": "user", "content": "..."}}, ...],
        "ground_truth": "expected answer" | ["call1()", "call2()"],
        "kwargs": {{}}  // optional extra args
    }}
    """
    try:
        body = event if isinstance(event, dict) else json.loads(event)

        raw_messages = body.get("messages", [])
        messages = [Message(**m) for m in raw_messages]
        ground_truth = body.get("ground_truth")
        kwargs = body.get("kwargs", {{}})

        result = _grader.grade(messages, ground_truth, **kwargs)

        return {{
            "statusCode": 200,
            "body": result.to_dict(),
        }}
    except Exception as e:
        return {{
            "statusCode": 500,
            "body": {{"error": str(e)}},
        }}
'''


def _build_deployment_package(grader_ref: str) -> bytes:
    """
    Build a Lambda deployment zip containing:
    - handler.py (generated from template)
    - The llm_eval_kit package
    - Third-party dependencies (pydantic, etc.) installed
      for the Lambda runtime platform
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    buf = io.BytesIO()

    # Find the llm_eval_kit package directory
    import llm_eval_kit
    pkg_dir = Path(llm_eval_kit.__file__).parent

    # Install dependencies into a temp dir for bundling.
    # Prefer uv for speed; fall back to pip if uv isn't available.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger.info("Installing dependencies into package...")

        pip_args = [
            "--target", str(tmp_path),
            "--platform", "manylinux2014_x86_64",
            "--implementation", "cp",
            "--python-version", "3.12",
            "--only-binary=:all:",
            "--quiet",
            "pydantic>=2.0.0",
            "pydantic-core",
            "annotated-types",
            "typing_extensions",
        ]

        uv_bin = shutil.which("uv")
        if uv_bin:
            subprocess.check_call(
                [uv_bin, "pip", "install"] + pip_args,
                stderr=subprocess.STDOUT,
            )
        else:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + pip_args,
                stderr=subprocess.STDOUT,
            )

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Write the handler
            handler_code = HANDLER_TEMPLATE.format(
                grader_ref=grader_ref,
            )
            zf.writestr("handler.py", handler_code)

            # 2. Bundle the llm_eval_kit package
            for file_path in pkg_dir.rglob("*.py"):
                arcname = str(
                    file_path.relative_to(pkg_dir.parent)
                )
                zf.writestr(arcname, file_path.read_text())

            # 3. Bundle pip-installed dependencies
            for file_path in tmp_path.rglob("*"):
                if file_path.is_file():
                    arcname = str(
                        file_path.relative_to(tmp_path)
                    )
                    zf.writestr(
                        arcname, file_path.read_bytes(),
                    )

    buf.seek(0)
    return buf.read()


def deploy_grader(
    grader_ref: str,
    config: Optional[DeployConfig] = None,
    config_path: Optional[str] = None,
) -> dict:
    """
    Deploy a grader as an AWS Lambda function.

    Args:
        grader_ref: Module path to the grader
            (e.g. "llm_eval_kit.graders.builtins.exact_match:exact_match_grader")
        config: DeployConfig instance (loaded from YAML if not provided)
        config_path: Path to config YAML file

    Returns:
        dict with deployment info (function_name, function_arn, etc.)
    """
    try:
        import boto3
        import botocore.exceptions
    except ImportError:
        raise ImportError(
            "boto3 not installed. Run: uv pip install -e \".[deploy]\""
        )

    if config is None:
        config = load_deploy_config(config_path)

    lc = config.aws.lambda_config
    region = config.aws.region
    profile = config.aws.profile

    # Build a session — supports named profiles, env vars, SSO,
    # instance roles, and the full default credential chain.
    session = boto3.Session(
        profile_name=profile,
        region_name=region,
    )

    # Validate credentials before doing any real work
    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        logger.info(
            "Authenticated as %s (account %s)",
            identity["Arn"], identity["Account"],
        )
    except botocore.exceptions.NoCredentialsError:
        raise RuntimeError(
            "No AWS credentials found. Set them up using one of:\n"
            "  1. aws configure            "
            "(writes ~/.aws/credentials)\n"
            "  2. aws configure sso        "
            "(SSO login)\n"
            "  3. Environment variables     "
            "(AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)\n"
            "  4. --profile flag or "
            "aws.profile in llm_eval_kit.yaml"
        )
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(
            f"AWS credential check failed: {e}\n"
            "Run 'aws sts get-caller-identity' to debug."
        )

    logger.info("Building deployment package for %s...", grader_ref)
    zip_bytes = _build_deployment_package(grader_ref)
    logger.info("Package size: %.1f KB", len(zip_bytes) / 1024)

    # Merge grader ref into environment
    env_vars = {**lc.environment, "GRADER_REF": grader_ref}

    client = session.client("lambda")

    # Check if function exists
    try:
        client.get_function(FunctionName=lc.function_name)
        exists = True
    except client.exceptions.ResourceNotFoundException:
        exists = False

    if exists:
        logger.info("Updating existing function: %s", lc.function_name)
        client.update_function_code(
            FunctionName=lc.function_name,
            ZipFile=zip_bytes,
        )
        # Wait for update to complete before updating config
        waiter = client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=lc.function_name)

        client.update_function_configuration(
            FunctionName=lc.function_name,
            Runtime=lc.runtime,
            Handler=lc.handler,
            Timeout=lc.timeout,
            MemorySize=lc.memory_size,
            Environment={"Variables": env_vars},
        )
        response = client.get_function(FunctionName=lc.function_name)
        arn = response["Configuration"]["FunctionArn"]
    else:
        if not lc.role_arn:
            raise ValueError(
                "role_arn is required to create a new Lambda function. "
                "Set it in llm_eval_kit.yaml under aws.lambda.role_arn "
                "or provide an existing function name to update."
            )
        logger.info("Creating new function: %s", lc.function_name)
        response = client.create_function(
            FunctionName=lc.function_name,
            Runtime=lc.runtime,
            Role=lc.role_arn,
            Handler=lc.handler,
            Code={"ZipFile": zip_bytes},
            Timeout=lc.timeout,
            MemorySize=lc.memory_size,
            Environment={"Variables": env_vars},
        )
        arn = response["FunctionArn"]

    result = {
        "function_name": lc.function_name,
        "function_arn": arn,
        "region": region,
        "grader_ref": grader_ref,
    }
    logger.info("Deployed: %s (%s)", lc.function_name, arn)
    return result


def deploy_reward_function(
    source_file: str,
    function_name: str,
    role_arn: str,
    handler: Optional[str] = None,
    runtime: str = "python3.12",
    timeout: int = 300,
    memory_size: int = 512,
    region: Optional[str] = None,
    profile: Optional[str] = None,
) -> dict:
    """
    Deploy a standalone reward function .py file as a Lambda.

    This is for zero-dependency reward functions that follow the
    Bedrock RFT batch contract (receive list, return list with
    id + aggregate_reward_score + reward_components).

    Unlike deploy_grader(), this does NOT bundle llm_eval_kit or
    pydantic — it just zips the single .py file and deploys it.

    Args:
        source_file: Path to the .py reward function file.
        function_name: Lambda function name.
        role_arn: IAM role ARN for the Lambda.
        handler: Lambda handler string. Defaults to
            "<module_name>.lambda_handler".
        runtime: Lambda runtime (default python3.12).
        timeout: Timeout in seconds (default 300).
        memory_size: Memory in MB (default 512).
        region: AWS region (default from env/config).
        profile: AWS profile name.

    Returns:
        dict with function_name, function_arn, region.
    """
    try:
        import boto3
        import botocore.exceptions
    except ImportError:
        raise ImportError(
            "boto3 not installed. "
            "Run: uv pip install -e \".[deploy]\""
        )

    from pathlib import Path as _Path

    src = _Path(source_file)
    if not src.is_file():
        raise FileNotFoundError(f"Reward function not found: {src}")

    module_name = src.stem
    if handler is None:
        handler = f"{module_name}.lambda_handler"

    # Build zip with just the single file
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{module_name}.py", src.read_text())
    buf.seek(0)
    zip_bytes = buf.read()

    logger.info(
        "Package: %s (%.1f KB)",
        module_name, len(zip_bytes) / 1024,
    )

    session = boto3.Session(
        profile_name=profile,
        region_name=region,
    )

    # Validate credentials
    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        logger.info(
            "Authenticated as %s", identity["Arn"],
        )
    except botocore.exceptions.NoCredentialsError:
        raise RuntimeError(
            "No AWS credentials found. "
            "Run 'aws configure' or set env vars."
        )

    client = session.client("lambda")

    try:
        client.get_function(FunctionName=function_name)
        exists = True
    except client.exceptions.ResourceNotFoundException:
        exists = False

    if exists:
        logger.info("Updating: %s", function_name)
        client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_bytes,
        )
        waiter = client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=function_name)
        client.update_function_configuration(
            FunctionName=function_name,
            Runtime=runtime,
            Handler=handler,
            Timeout=timeout,
            MemorySize=memory_size,
        )
        resp = client.get_function(FunctionName=function_name)
        arn = resp["Configuration"]["FunctionArn"]
    else:
        logger.info("Creating: %s", function_name)
        resp = client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": zip_bytes},
            Timeout=timeout,
            MemorySize=memory_size,
        )
        arn = resp["FunctionArn"]

    # Wait for function to be active
    waiter = client.get_waiter("function_active_v2")
    waiter.wait(FunctionName=function_name)

    result = {
        "function_name": function_name,
        "function_arn": arn,
        "region": session.region_name,
    }
    logger.info("Deployed: %s (%s)", function_name, arn)
    return result
