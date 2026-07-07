"""CPU-only tests for the MLX backend's per-model coordinate profile.

No model load: exercises the pure profile-detection and denormalization
helpers that let a stock Qwen3-VL (0-1000 normalized grounding) drop into the
same backend as the r64 fine-tune (absolute image pixels).
"""

from __future__ import annotations

from ais5.pointcast.backends.mlx_backend import _coord_profile_for, _to_image_pixels
from ais5.prompt.action import ParsedAction


class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type


def test_profile_detection_by_model_type():
    assert _coord_profile_for(_Cfg("qwen3_vl")) == "norm1000"
    assert _coord_profile_for(_Cfg("qwen3_vl_moe")) == "norm1000"
    assert _coord_profile_for(_Cfg("qwen2_5_vl")) == "absolute"
    assert _coord_profile_for({"model_type": "qwen3_vl"}) == "norm1000"
    assert _coord_profile_for({"model_type": "qwen2_5_vl"}) == "absolute"
    assert _coord_profile_for({}) == "absolute"
    assert _coord_profile_for(None) == "absolute"


def test_norm1000_point_denormalizes_to_image_pixels():
    parsed = ParsedAction(point=(500.0, 250.0), raw="<click>500, 250</click>", parser="qwen-click")
    out = _to_image_pixels(parsed, (1512, 982), "norm1000")
    assert out.point == (756.0, 245.5)
    assert out.parser == "qwen-click"  # everything else untouched


def test_norm1000_bbox_denormalizes_too():
    parsed = ParsedAction(point=(500.0, 500.0), bbox=(100.0, 200.0, 300.0, 400.0), raw="", parser="box-tag")
    out = _to_image_pixels(parsed, (1000, 500), "norm1000")
    assert out.bbox == (100.0, 100.0, 300.0, 200.0)


def test_absolute_profile_is_passthrough():
    parsed = ParsedAction(point=(756.0, 245.0), raw="", parser="qwen-click")
    out = _to_image_pixels(parsed, (1512, 982), "absolute")
    assert out is parsed


def test_none_point_is_passthrough():
    parsed = ParsedAction(point=None, raw="no idea", parser="none")
    assert _to_image_pixels(parsed, (1512, 982), "norm1000") is parsed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} coord-profile tests passed")
