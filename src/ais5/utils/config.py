"""YAML config loader with shallow ${VAR} expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)

        return _VAR_RE.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand `${VAR}` / `${VAR:-default}` references."""
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)
