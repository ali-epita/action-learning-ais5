"""Top-1 click accuracy on bounding-box ground truth.

This is the primary metric across all three modeling tasks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Bbox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class ClickResult:
    sample_id: str
    pred: Point | None
    bbox: Bbox
    correct: bool
    benchmark: str
    target_type: str | None = None
    ui_type: str | None = None
    target_relative_area: float | None = None
    raw_response: str = ""
    latency_ms: float | None = None


def point_in_bbox(point: Point, bbox: Bbox, *, tolerance: float = 0.0) -> bool:
    """True iff `point` is inside `bbox` (inclusive), with optional pixel slack."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - tolerance) <= x <= (x2 + tolerance) and (y1 - tolerance) <= y <= (y2 + tolerance)


def click_accuracy(results: Iterable[ClickResult]) -> float:
    """Fraction of `results` that landed inside the gold bbox.

    `pred=None` (parser failed) counts as incorrect — that's intentional, since
    a parser miss is still a model failure under the "same action format" rule.
    """
    rs = list(results)
    if not rs:
        return 0.0
    return sum(1 for r in rs if r.correct) / len(rs)
