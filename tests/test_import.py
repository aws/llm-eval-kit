from textwrap import dedent
import pytest

def test_import() -> None:
    import llm_eval_kit  # type: ignore # noqa: F401
