"""End-to-end test for the crop-then-click policy with a fake model.

No GPU/network needed — we wire up a `GUIModel` whose predict() emits a
deterministic point so we can exercise the geometry of the crop wrapper.
"""

from __future__ import annotations

from PIL import Image

from ais5.models.base import GUIModel, ModelOutput
from ais5.prompt.action import ParsedAction
from ais5.tile.crop_then_click import CropConfig, crop_then_click


class FakeModel(GUIModel):
    name = "fake"
    param_count_b = 0.0

    def __init__(self, points: list[tuple[float, float] | None]) -> None:
        self._queue = list(points)

    def predict(self, image, instruction, **kwargs):
        point = self._queue.pop(0)
        return ModelOutput(
            text="<fake>",
            parsed=ParsedAction(point=point, raw="", parser="fake"),
        )


def test_crop_then_click_translates_back_to_original_coords():
    # A 1000×1000 image, coarse predicts (700, 700), refined predicts (50, 50)
    # in the crop. With crop_size=200, crop box should be (600,600,800,800)
    # → final point = (650, 650).
    image = Image.new("RGB", (1000, 1000), color="white")
    model = FakeModel([(700, 700), (50, 50)])
    out = crop_then_click(model, image, "click", cfg=CropConfig(crop_size=200))
    assert out.parsed.point == (650.0, 650.0)
    assert out.metadata["crop_box"] == (600, 600, 800, 800)
    assert out.metadata["refined_coord_frame"] == "crop-local"


def test_crop_then_click_keeps_full_image_refined_coords():
    image = Image.new("RGB", (1000, 1000), color="white")
    model = FakeModel([(700, 700), (900, 100)])
    out = crop_then_click(model, image, "click", cfg=CropConfig(crop_size=200))
    assert out.parsed.point == (900, 100)
    assert out.metadata["crop_box"] == (600, 600, 800, 800)
    assert out.metadata["refined_coord_frame"] == "full-image"


def test_crop_then_click_returns_coarse_when_stage1_fails():
    image = Image.new("RGB", (1000, 1000))
    model = FakeModel([None])
    out = crop_then_click(model, image, "click")
    assert out.parsed.point is None


def test_crop_then_click_falls_back_when_stage2_fails():
    image = Image.new("RGB", (1000, 1000))
    model = FakeModel([(500, 500), None])  # coarse OK, refined None
    out = crop_then_click(model, image, "click")
    # Falls back to coarse; original-image coords are unchanged.
    assert out.parsed.point == (500, 500)


def test_crop_box_clamps_to_image():
    image = Image.new("RGB", (200, 200))
    model = FakeModel([(180, 180), (10, 10)])
    out = crop_then_click(model, image, "click", cfg=CropConfig(crop_size=128))
    x1, y1, x2, y2 = out.metadata["crop_box"]
    assert (x1, y1, x2, y2) == (72, 72, 200, 200)
