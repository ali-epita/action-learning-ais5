"""Benchmark registry — `load_benchmark(name, ...)` returns a sample iterator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .osworld_g import load_osworld_g
from .screenspot import load_screenspot_pro, load_screenspot_v2
from .types import GroundingSample

_BENCHMARKS = {
    "screenspot-v2": load_screenspot_v2,
    "screenspot-pro": load_screenspot_pro,
    "osworld-g": load_osworld_g,
}


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS.keys())


def load_benchmark(name: str, **kwargs: Any) -> Iterator[GroundingSample]:
    if name not in _BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Available: {', '.join(list_benchmarks())}"
        )
    return _BENCHMARKS[name](**kwargs)
