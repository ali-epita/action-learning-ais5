"""Pareto-front extraction for accuracy-vs-X trade-offs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    accuracy: float
    cost: float  # latency, VRAM, parameter count, etc. — lower is better

    def dominates(self, other: ParetoPoint) -> bool:
        better_or_equal = self.accuracy >= other.accuracy and self.cost <= other.cost
        strictly_better = self.accuracy > other.accuracy or self.cost < other.cost
        return better_or_equal and strictly_better


def pareto_front(points: Iterable[ParetoPoint]) -> list[ParetoPoint]:
    """Return the non-dominated subset of `points`, sorted by ascending cost."""
    points = list(points)
    front = []
    for p in points:
        if not any(q.dominates(p) for q in points if q is not p):
            front.append(p)
    return sorted(front, key=lambda p: p.cost)
