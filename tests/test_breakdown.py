"""Tests for the per-slice accuracy breakdowns."""

from __future__ import annotations

from ais5.eval.breakdown import bucket_for_area, by_target_size, by_ui_type
from ais5.eval.click import ClickResult


def _r(*, correct: bool, area: float | None = None, ui: str | None = None) -> ClickResult:
    return ClickResult(
        sample_id="x",
        pred=(0, 0),
        bbox=(0, 0, 1, 1),
        correct=correct,
        benchmark="b",
        target_relative_area=area,
        ui_type=ui,
    )


def test_bucket_thresholds():
    assert bucket_for_area(5e-5).startswith("xs")
    assert bucket_for_area(5e-4).startswith("s ")
    assert bucket_for_area(5e-3).startswith("m ")
    assert bucket_for_area(5e-2).startswith("l ")
    assert bucket_for_area(None) == "unknown"


def test_by_target_size_aggregates():
    rows = [
        _r(correct=True, area=5e-5),
        _r(correct=False, area=5e-5),
        _r(correct=True, area=5e-2),
    ]
    df = by_target_size(rows)
    assert {"size", "correct", "total", "accuracy"} <= set(df.columns)
    xs = df[df["size"].str.startswith("xs")].iloc[0]
    assert xs["total"] == 2
    assert xs["accuracy"] == 0.5


def test_by_ui_type_handles_missing():
    rows = [_r(correct=True, ui=None), _r(correct=False, ui="web")]
    df = by_ui_type(rows)
    assert set(df["ui"]) == {"unknown", "web"}
