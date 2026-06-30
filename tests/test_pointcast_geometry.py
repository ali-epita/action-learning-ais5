"""CPU-only tests for PointCast coordinate mapping + locator (no model, no Qt)."""

from __future__ import annotations

from ais5.pointcast.geometry import CoordinateMapper, compute_ground_scale
from ais5.pointcast.locator import confirmation_phrase, describe_region


def test_ground_scale_downscales_only():
    assert compute_ground_scale(3024, 1964, 1512) == 1512 / 3024  # 0.5
    assert compute_ground_scale(1280, 800, 1512) == 1.0  # already fits → no upscale
    assert compute_ground_scale(3024, 1964, 0) == 1.0  # disabled
    assert compute_ground_scale(3024, 1964, None) == 1.0


def test_retina_downscale_center_maps_to_center():
    # 3024x1964 physical, 1512x982 logical (Retina 2x), grounded at 0.5 -> 1512x982.
    m = CoordinateMapper(capture_size=(3024, 1964), logical_size=(1512, 982), ground_scale=0.5)
    assert m.retina_scale == 2.0
    assert m.grounding_size == (1512, 982)
    # grounding-image center -> logical center
    lx, ly = m.ground_to_logical(756, 491)
    assert abs(lx - 756) < 1e-6 and abs(ly - 491) < 1e-6
    # top-left and bottom-right corners
    assert m.ground_to_logical(0, 0) == (0.0, 0.0)
    assert m.ground_to_logical(1512, 982) == (1512.0, 982.0)


def test_non_retina_no_downscale_is_identity():
    m = CoordinateMapper(capture_size=(1280, 800), logical_size=(1280, 800), ground_scale=1.0)
    assert m.retina_scale == 1.0
    assert m.ground_to_logical(640, 400) == (640.0, 400.0)


def test_monitor_offset_applied():
    # secondary monitor at physical x-offset 1512 (logical scale 1x here)
    m = CoordinateMapper(
        capture_size=(1920, 1080), logical_size=(1920, 1080), ground_scale=1.0,
        offset_physical=(1512, 0),
    )
    assert m.ground_to_logical(10, 20) == (1522.0, 20.0)


def test_physical_overlay_coords():
    m = CoordinateMapper(capture_size=(3024, 1964), logical_size=(1512, 982), ground_scale=0.5)
    # grounding point -> physical device px (what a device-pixel overlay paints in)
    assert m.ground_to_physical(756, 491) == (1512.0, 982.0)


def test_describe_region_grid():
    w, h = 1200, 900
    assert describe_region(50, 50, w, h) == "top left"
    assert describe_region(600, 450, w, h) == "center"
    assert describe_region(1150, 850, w, h) == "bottom right"
    assert describe_region(600, 50, w, h) == "top center"
    assert describe_region(1150, 450, w, h) == "center right"


def test_confirmation_phrase():
    p = confirmation_phrase("the Send button", 1150, 850, 1200, 900)
    assert "Send button" in p and "bottom right" in p
    assert confirmation_phrase("", 600, 450, 1200, 900) == "Clicking center."


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} geometry/locator tests passed")
