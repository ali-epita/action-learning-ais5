"""Coverage for every dialect ais5.prompt.parse_click should understand."""

from __future__ import annotations

import pytest

from ais5.prompt.action import parse_click


def test_qwen_click_tag():
    p = parse_click("<click>123, 456</click>")
    assert p.point == (123.0, 456.0)
    assert p.parser == "qwen-click"


def test_qwen_click_tag_floats_and_spaces():
    p = parse_click("Sure! <click>  12.5  ,  640.0  </click>")
    assert p.point == (12.5, 640.0)


def test_box_tag_returns_centroid():
    p = parse_click("<box>10, 20, 30, 40</box>")
    assert p.point == (20.0, 30.0)
    assert p.bbox == (10.0, 20.0, 30.0, 40.0)
    assert p.parser == "box-tag"


def test_paligemma_loc_tokens_with_image_size():
    # row=512, col=512 in a 1024-bin grid → roughly the center of a 1000x1000 image
    p = parse_click("<loc0512><loc0512>", image_size=(1000, 1000))
    assert p.point is not None
    x, y = p.point
    assert abs(x - 500.0) < 1.0
    assert abs(y - 500.0) < 1.0
    assert p.parser == "paligemma-loc"


def test_paligemma_loc_tokens_without_image_size_falls_through():
    # No image_size → can't denormalize; parser should fall through
    p = parse_click("<loc0123><loc0456>")
    # falls through to none unless another regex catches it
    assert p.parser != "paligemma-loc"


def test_json_object_with_x_y():
    p = parse_click('Output: {"action": "click", "x": 200, "y": 300}')
    assert p.point == (200.0, 300.0)
    assert p.parser == "json-object"


def test_json_object_nested_action():
    p = parse_click('{"reasoning": "click save", "action": {"x": 50, "y": 75}}')
    assert p.point == (50.0, 75.0)


def test_pixel_tuple_fallback():
    p = parse_click("The element is at (456, 123).")
    assert p.point == (456.0, 123.0)
    assert p.parser == "pixel-tuple"


def test_unparseable_returns_none():
    p = parse_click("I have no idea what to click")
    assert p.point is None
    assert p.parser == "none"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<click>0, 0</click>", (0.0, 0.0)),
        ("<click>-5, 10</click>", (-5.0, 10.0)),  # negatives accepted; eval rejects them anyway
    ],
)
def test_edge_coordinates(text, expected):
    assert parse_click(text).point == expected
