"""CLI entry point for llm-eval-kit."""
import argparse
import json
import sys
from pathlib import Path

from llm_eval_kit.datasets.loader import load_bfcl, load_jsonl
from llm_eval_kit.execution.pipeline import EvalPipeline
from llm_eval_kit.graders.registry import default_registry
from llm_eval_kit.models.datasets import EvalSample
from llm_eval_kit.utils.module_loader import load_function

# Top-level imports so graders are registered once
import llm_eval_kit.graders  # noqa: F401


def _cmd_evaluate(args):
    if args.grader:
        try:
            g = default_registry.get(args.grader)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.grader_path:
        try:
            g = load_function(args.grader_path)
        except (ImportError, AttributeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: --grader or --grader-path required",
              file=sys.stderr)
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"Error: not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    if args.format == "bfcl":
        ds = load_bfcl(args.data, max_samples=args.max_samples)
    else:
        ds = load_jsonl(args.data, max_samples=args.max_samples)

    report = EvalPipeline(g, ds).run_with_report()
    print(report.summary())
    if args.output:
        report.to_jsonl(args.output)
        print(f"Results written to {args.output}")


def _cmd_list_graders():
    for name in default_registry.list_graders():
        g = default_registry.get(name)
        print(f"  {name}: {g.description}")


def _cmd_validate(args):
    if not Path(args.data).exists():
        print(f"Error: not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    errors = []
    count = 0
    with open(args.data) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                EvalSample(**json.loads(line))
                count += 1
            except Exception as e:
                errors.append((ln, str(e)))
    if errors:
        for ln, err in errors:
            print(f"  Line {ln}: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"Valid: {count} samples")


def _cmd_deploy(args):
    """Deploy a grader as an AWS Lambda reward function."""
    try:
        from llm_eval_kit.deploy.lambda_deploy import deploy_grader
        from llm_eval_kit.deploy.config import load_deploy_config
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install deploy extras: pip install llm-eval-kit[deploy]",
              file=sys.stderr)
        sys.exit(1)

    # Resolve grader reference
    if args.grader:
        grader_ref = (
            f"llm_eval_kit.graders.builtins.{args.grader}"
            f":{args.grader}_grader"
        )
    elif args.grader_path:
        grader_ref = args.grader_path
    else:
        print("Error: --grader or --grader-path required", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_deploy_config(args.config)
        # CLI flags override config file / env vars
        if args.profile:
            config.aws.profile = args.profile
        if args.region:
            config.aws.region = args.region
        if args.role_arn:
            config.aws.lambda_config.role_arn = args.role_arn
        if args.function_name:
            config.aws.lambda_config.function_name = (
                args.function_name
            )
        result = deploy_grader(grader_ref, config=config)
        print(f"Deployed: {result['function_name']}")
        print(f"  ARN:    {result['function_arn']}")
        print(f"  Region: {result['region']}")
        print(f"  Grader: {result['grader_ref']}")
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Deploy failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Entry point for llm-eval-kit CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-eval-kit",
        description="LLM Evaluation Toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    ep = sub.add_parser("evaluate", help="Run evaluation")
    ep.add_argument("--grader", type=str)
    ep.add_argument("--grader-path", type=str)
    ep.add_argument("--data", required=True)
    ep.add_argument("--output", type=str)
    ep.add_argument("--max-samples", type=int)
    ep.add_argument("--format", choices=["bfcl", "jsonl"],
                    default="jsonl")

    sub.add_parser("list-graders", help="List graders")

    vp = sub.add_parser("validate", help="Validate dataset")
    vp.add_argument("--data", required=True)

    dp = sub.add_parser("deploy", help="Deploy grader as Lambda")
    dp.add_argument("--grader", type=str,
                    help="Built-in grader name (e.g. exact_match)")
    dp.add_argument("--grader-path", type=str,
                    help="Module path (e.g. my_module:my_grader)")
    dp.add_argument("--config", type=str,
                    help="Path to llm_eval_kit.yaml config file")
    dp.add_argument("--profile", type=str,
                    help="AWS profile name (from ~/.aws/credentials)")
    dp.add_argument("--region", type=str,
                    help="AWS region (overrides config/env)")
    dp.add_argument("--role-arn", type=str,
                    help="IAM role ARN for the Lambda function")
    dp.add_argument("--function-name", type=str,
                    help="Lambda function name (default: "
                         "llm-eval-reward-function)")

    args = parser.parse_args()
    if args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "list-graders":
        _cmd_list_graders()
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "deploy":
        _cmd_deploy(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
