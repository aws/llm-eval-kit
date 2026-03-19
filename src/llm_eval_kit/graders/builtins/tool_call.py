"""
Tool call grader — BFCL-style AST comparison of function calls.

Parses function call strings like `func_name(param1=value1, param2="str")`
using Python's ast module, then compares structurally with type coercion.
"""
import ast
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from llm_eval_kit.graders.decorator import grader
from llm_eval_kit.models.results import EvaluateResult, MetricResult
from ._helpers import get_last_assistant_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function call parser
# ---------------------------------------------------------------------------

@dataclass
class ParsedCall:
    """Structured representation of a parsed function call."""
    func_name: str
    params: Dict[str, Any]


def parse_function_call(call_str: str) -> ParsedCall:
    """
    Parse 'func_name(param1=value1, param2="str")' into a ParsedCall.

    Uses ast.parse in eval mode to safely parse the expression,
    then extracts function name and keyword arguments.
    """
    call_str = call_str.strip()
    try:
        tree = ast.parse(call_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cannot parse function call: {call_str!r}") from e

    if not isinstance(tree.body, ast.Call):
        raise ValueError(f"Expression is not a function call: {call_str!r}")

    call_node = tree.body

    # Extract function name (handles simple names like func_name)
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        # Handle dotted names like module.func_name
        parts = []
        node = call_node.func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        func_name = ".".join(reversed(parts))
    else:
        raise ValueError(f"Unsupported function call format: {call_str!r}")

    # Extract keyword arguments
    params: Dict[str, Any] = {}

    # Handle positional args (convert to indexed params)
    for i, arg in enumerate(call_node.args):
        try:
            params[f"_arg{i}"] = ast.literal_eval(arg)
        except (ValueError, TypeError):
            params[f"_arg{i}"] = ast.dump(arg)

    # Handle keyword args
    for kw in call_node.keywords:
        if kw.arg is None:
            continue  # **kwargs expansion, skip
        try:
            params[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            # Fall back to string representation for complex expressions
            params[kw.arg] = ast.dump(kw.value)

    return ParsedCall(func_name=func_name, params=params)


def format_function_call(parsed: ParsedCall) -> str:
    """Pretty-print a ParsedCall back to a function call string."""
    param_strs = [f"{k}={repr(v)}" for k, v in sorted(parsed.params.items())]
    return f"{parsed.func_name}({', '.join(param_strs)})"


# ---------------------------------------------------------------------------
# Type-coerced value comparison
# ---------------------------------------------------------------------------

def _try_coerce_match(a: Any, b: Any) -> bool:
    """Try to coerce `a` to the type of `b` and compare."""
    if isinstance(a, str) and isinstance(b, bool):
        if a.lower() in ("true", "false"):
            return (a.lower() == "true") == b
        return False
    if isinstance(a, str) and isinstance(b, int) and not isinstance(b, bool):
        try:
            return int(a) == b
        except (ValueError, TypeError):
            return False
    if isinstance(a, str) and isinstance(b, float):
        try:
            return float(a) == b
        except (ValueError, TypeError):
            return False
    return False


def values_match(predicted: Any, expected: Any) -> bool:
    """Compare two values with type coercion."""
    if predicted == expected:
        return True
    # Try coercion in both directions
    if _try_coerce_match(predicted, expected):
        return True
    if _try_coerce_match(expected, predicted):
        return True
    # Recursive comparison for lists
    if isinstance(predicted, list) and isinstance(expected, list):
        if len(predicted) != len(expected):
            return False
        return all(values_match(p, e) for p, e in zip(predicted, expected))
    # Recursive comparison for dicts
    if isinstance(predicted, dict) and isinstance(expected, dict):
        if set(predicted.keys()) != set(expected.keys()):
            return False
        return all(
            values_match(predicted[k], expected[k]) for k in expected
        )
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compare_single_call(predicted: ParsedCall, expected: ParsedCall) -> dict:
    """Compare one predicted call against one expected call."""
    func_match = predicted.func_name == expected.func_name

    expected_keys = set(expected.params.keys())
    predicted_keys = set(predicted.params.keys())
    common_keys = expected_keys & predicted_keys

    if expected_keys:
        param_name_acc = len(common_keys) / len(expected_keys)
    else:
        param_name_acc = 1.0 if not predicted_keys else 0.0

    value_matches = sum(
        1 for k in common_keys
        if values_match(predicted.params[k], expected.params[k])
    )
    param_value_acc = value_matches / len(expected_keys) if expected_keys else 1.0

    overall = (
        (1.0 if func_match else 0.0) * 0.33
        + param_name_acc * 0.33
        + param_value_acc * 0.34
    )
    return {
        "func_name_match": func_match,
        "param_name_accuracy": param_name_acc,
        "param_value_accuracy": param_value_acc,
        "overall": overall,
    }


def _split_calls(text: str) -> List[str]:
    """
    Split a string that may contain multiple function calls.
    Handles newline-separated or list-formatted calls.
    """
    text = text.strip()
    # If it looks like a Python list, try to parse individual calls
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Split on newlines or comma-separated top-level calls
    calls = []
    depth = 0
    current: List[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "\n" and depth == 0:
            chunk = "".join(current).strip()
            if chunk:
                calls.append(chunk)
            current = []
        else:
            current.append(char)
    chunk = "".join(current).strip()
    if chunk:
        calls.append(chunk)

    # Clean trailing commas
    return [c.rstrip(",").strip() for c in calls if c.strip()]


# ---------------------------------------------------------------------------
# The grader
# ---------------------------------------------------------------------------

@grader(
    name="tool_call",
    description="BFCL-style AST comparison of function calls",
)
def tool_call_grader(messages, ground_truth, **kwargs):
    """
    Compare predicted function calls against ground truth.

    ground_truth: str or List[str] of function call strings
    messages: last assistant message content contains predicted call(s)
    """
    predicted_str = get_last_assistant_content(messages)
    if predicted_str is None:
        return EvaluateResult(
            score=0.0, is_valid=False, reason="No assistant message"
        )

    # Normalize ground_truth to list of strings
    if isinstance(ground_truth, str):
        gt_strs = [ground_truth]
    elif isinstance(ground_truth, list):
        gt_strs = [str(g) for g in ground_truth]
    else:
        return EvaluateResult(
            score=0.0,
            is_valid=False,
            reason=f"Unexpected ground_truth type: {type(ground_truth)}",
        )

    # Parse predicted calls
    try:
        predicted_calls = [parse_function_call(s) for s in _split_calls(predicted_str)]
    except (ValueError, SyntaxError) as e:
        return EvaluateResult(
            score=0.0, is_valid=False, reason=f"Parse error (predicted): {e}"
        )

    # Parse expected calls
    try:
        expected_calls = [parse_function_call(s) for s in gt_strs]
    except (ValueError, SyntaxError) as e:
        return EvaluateResult(
            score=0.0, is_valid=False, reason=f"Parse error (ground truth): {e}"
        )

    # Compare by position
    comparisons = []
    for pred, exp in zip(predicted_calls, expected_calls):
        comparisons.append(compare_single_call(pred, exp))

    n = max(len(expected_calls), len(predicted_calls), 1)
    avg_overall = sum(c["overall"] for c in comparisons) / n if comparisons else 0.0

    fn_acc = sum(1.0 if c["func_name_match"] else 0.0 for c in comparisons) / n
    pn_acc = sum(c["param_name_accuracy"] for c in comparisons) / n
    pv_acc = sum(c["param_value_accuracy"] for c in comparisons) / n

    return EvaluateResult(
        score=avg_overall,
        reason=f"Matched {len(comparisons)}/{n} function calls",
        metrics={
            "function_name_accuracy": MetricResult(
                score=fn_acc,
                reason="Fraction of calls with correct function name",
            ),
            "parameter_name_accuracy": MetricResult(
                score=pn_acc,
                reason="Average parameter name accuracy across calls",
            ),
            "parameter_value_accuracy": MetricResult(
                score=pv_acc,
                reason="Average parameter value accuracy across calls",
            ),
        },
    )
