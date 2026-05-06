"""Per-slice accuracy breakdowns used by all three tasks' deeper analyses.

Task 1: by UI type (web/desktop/mobile) — error analysis
Task 2: by target size — failure mode for small-icon high-res screenshots
Task 3: by latency / VRAM — Pareto fronts (in ais5.bench)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from .click import ClickResult

# Target-area buckets in fraction-of-image units. Tuned for ScreenSpot-Pro:
# professional UIs have many sub-1e-3 targets that fail without zoom-in.
SIZE_BUCKETS: list[tuple[str, float, float]] = [
    ("xs (<0.01%)", 0.0, 1e-4),
    ("s (0.01-0.1%)", 1e-4, 1e-3),
    ("m (0.1-1%)", 1e-3, 1e-2),
    ("l (>=1%)", 1e-2, 1.01),
]


def bucket_for_area(area: float | None) -> str:
    if area is None:
        return "unknown"
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def _aggregate(results: Iterable[ClickResult], key: str) -> pd.DataFrame:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        if key == "size":
            label = bucket_for_area(r.target_relative_area)
        elif key == "ui":
            label = r.ui_type or "unknown"
        elif key == "type":
            label = r.target_type or "unknown"
        else:
            raise ValueError(f"Unknown breakdown key: {key!r}")
        counts[label][0] += int(r.correct)
        counts[label][1] += 1
    rows = [
        {key: label, "correct": c, "total": t, "accuracy": c / t if t else 0.0}
        for label, (c, t) in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def by_target_size(results: Iterable[ClickResult]) -> pd.DataFrame:
    return _aggregate(results, "size")


def by_ui_type(results: Iterable[ClickResult]) -> pd.DataFrame:
    return _aggregate(results, "ui")


def by_target_type(results: Iterable[ClickResult]) -> pd.DataFrame:
    return _aggregate(results, "type")
