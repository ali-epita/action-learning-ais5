"""Filesystem helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_path(*parts: str, run_id: str | None = None) -> Path:
    """Return a path under ./results/, creating its parent dirs.

    A `run_id` like "2026-05-06T14-30-12" can be passed to namespace outputs.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    full = RESULTS_ROOT.joinpath(run_id, *parts)
    ensure_dir(full.parent)
    return full


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent, default=_default)


def write_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
