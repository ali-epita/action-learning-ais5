"""Tests for the Pareto-front extractor (Task 3)."""

from __future__ import annotations

from ais5.bench.pareto import ParetoPoint, pareto_front


def test_dominates_strictly_better_on_both():
    a = ParetoPoint("a", accuracy=0.9, cost=100)
    b = ParetoPoint("b", accuracy=0.8, cost=200)
    assert a.dominates(b)
    assert not b.dominates(a)


def test_dominates_equal_one_strictly_better():
    a = ParetoPoint("a", accuracy=0.9, cost=100)
    b = ParetoPoint("b", accuracy=0.9, cost=200)
    assert a.dominates(b)
    assert not b.dominates(a)


def test_pareto_front_keeps_non_dominated():
    pts = [
        ParetoPoint("cheap-bad", accuracy=0.5, cost=50),
        ParetoPoint("balanced", accuracy=0.7, cost=100),
        ParetoPoint("expensive-good", accuracy=0.9, cost=400),
        ParetoPoint("dominated", accuracy=0.6, cost=200),  # worse than balanced & still costlier
    ]
    front = pareto_front(pts)
    labels = [p.label for p in front]
    assert "dominated" not in labels
    assert "balanced" in labels
    # Sorted ascending by cost
    assert front == sorted(front, key=lambda p: p.cost)


def test_pareto_front_empty():
    assert pareto_front([]) == []
