"""Tests for click_accuracy and the bbox containment check."""

from __future__ import annotations

from ais5.eval.click import ClickResult, click_accuracy, point_in_bbox


def test_point_inside_bbox():
    assert point_in_bbox((50, 50), (10, 10, 100, 100))


def test_point_outside_bbox():
    assert not point_in_bbox((150, 150), (10, 10, 100, 100))


def test_point_on_bbox_edge_is_inside():
    assert point_in_bbox((10, 10), (10, 10, 100, 100))
    assert point_in_bbox((100, 100), (10, 10, 100, 100))


def test_tolerance_extends_box():
    assert not point_in_bbox((105, 50), (10, 10, 100, 100))
    assert point_in_bbox((105, 50), (10, 10, 100, 100), tolerance=10)


def _make(correct: bool) -> ClickResult:
    return ClickResult(
        sample_id="x",
        pred=(0, 0),
        bbox=(0, 0, 1, 1),
        correct=correct,
        benchmark="b",
    )


def test_click_accuracy_basic():
    rs = [_make(True), _make(True), _make(False), _make(True)]
    assert click_accuracy(rs) == 0.75


def test_click_accuracy_empty():
    assert click_accuracy([]) == 0.0


def test_click_accuracy_failed_parse_counts_as_wrong():
    # A None prediction means parsing failed; pre-built ClickResult.correct=False
    failed = ClickResult(
        sample_id="x",
        pred=None,
        bbox=(0, 0, 10, 10),
        correct=False,
        benchmark="b",
    )
    assert click_accuracy([failed, _make(True)]) == 0.5
