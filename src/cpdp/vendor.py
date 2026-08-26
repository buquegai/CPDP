from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_module(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Vendor module not found: {path}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load vendor module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

