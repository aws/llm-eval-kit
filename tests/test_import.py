from textwrap import dedent
import pytest

def test_import() -> None:
    import nova_custom_evaluation_sdk  # type: ignore # noqa: F401
