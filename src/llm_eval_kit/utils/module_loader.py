"""Dynamic module/function loader — load Python objects from string paths."""
import importlib
from typing import Any


def load_function(path: str) -> Any:
    """
    Load a Python object from a string path.

    Supports:
      - "module.submodule:func_name" (colon format, preferred)
      - "module.submodule.func_name" (dot format, last component is attribute)
    """
    if ":" in path:
        module_path, attr_name = path.split(":", 1)
    else:
        parts = path.rsplit(".", 1)
        if len(parts) < 2:
            raise ImportError(
                f"Invalid path format: {path!r}. "
                f"Expected 'module.path:func' or 'module.path.func'"
            )
        module_path, attr_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}': {e}"
        ) from e

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        )
    return getattr(module, attr_name)
