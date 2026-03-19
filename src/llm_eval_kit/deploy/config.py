"""
Deployment configuration — YAML-based config for AWS Lambda deployment.

Reads from llm_eval_kit.yaml or a user-specified path.
Falls back to environment variables for AWS credentials.
"""
import os
import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "llm_eval_kit.yaml"


class LambdaConfig(BaseModel):
    """Lambda function configuration."""
    function_name: str = "llm-eval-reward-function"
    runtime: str = "python3.12"
    handler: str = "handler.lambda_handler"
    timeout: int = 60
    memory_size: int = 256
    role_arn: Optional[str] = None
    layers: list = Field(default_factory=list)
    environment: Dict[str, str] = Field(default_factory=dict)


class AWSConfig(BaseModel):
    """AWS account configuration."""
    region: str = "us-east-1"
    account_id: Optional[str] = None
    profile: Optional[str] = None
    lambda_config: LambdaConfig = Field(
        default_factory=LambdaConfig,
        alias="lambda",
    )

    model_config = {"populate_by_name": True}


class DeployConfig(BaseModel):
    """Top-level deployment configuration."""
    aws: AWSConfig = Field(default_factory=AWSConfig)


def load_deploy_config(
    config_path: Optional[str] = None,
) -> DeployConfig:
    """
    Load deployment config from YAML file.

    Search order:
    1. Explicit config_path argument
    2. llm_eval_kit.yaml in current directory
    3. Walk up parent directories
    4. Fall back to defaults + env vars
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML not installed. Run: uv pip install -e \".[deploy]\""
        )

    # Find config file
    if config_path is None:
        config_path = _find_config_file()

    if config_path and os.path.isfile(config_path):
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        config = DeployConfig(**raw)
    else:
        logger.info("No config file found, using defaults")
        config = DeployConfig()

    # Override from environment variables
    if not config.aws.account_id:
        config.aws.account_id = os.environ.get("AWS_ACCOUNT_ID")
    if os.environ.get("AWS_DEFAULT_REGION"):
        config.aws.region = os.environ["AWS_DEFAULT_REGION"]
    if os.environ.get("AWS_REGION"):
        config.aws.region = os.environ["AWS_REGION"]
    if not config.aws.profile:
        config.aws.profile = os.environ.get("AWS_PROFILE")

    return config


def _find_config_file() -> Optional[str]:
    """Walk up from CWD looking for llm_eval_kit.yaml."""
    current = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(current, CONFIG_FILE_NAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
