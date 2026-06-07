"""Benchmark registry — `load_benchmark(name, ...)` returns a sample iterator."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
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


def iter_benchmark(
    name: str, *, limit: int | None = None, **kwargs: Any
) -> Iterator[GroundingSample]:
    """Lazily yield benchmark samples, optionally capped at `limit`.

    Keeps one decoded image in memory at a time. Use this instead of
    `list(load_benchmark(name))` in notebooks: materializing a whole benchmark
    (e.g. ScreenSpot-Pro's high-resolution screenshots) holds every decoded
    image in RAM at once and OOMs Colab.
    """
    stream: Iterator[GroundingSample] = load_benchmark(name, **kwargs)
    if limit is not None:
        stream = islice(stream, limit)
    yield from stream
