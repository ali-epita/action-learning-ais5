"""Two-stage 'glance-then-focus' policy from GTA1 / Ferret-UI Lite (paper #1, #7).

    1. Predict on the full screenshot → coarse point.
    2. Crop a window of `crop_size` centered on that point.
    3. Predict on the crop → refined point.
    4. Translate the refined point back to original-image coordinates.

If the first stage fails to parse a point, we fall back to the full-image
prediction (no refinement). The wrapper is deliberately model-agnostic — it
just needs anything implementing `GUIModel.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image

from ..models.base import GUIModel, ModelOutput
from ..prompt.action import ParsedAction


@dataclass
class CropConfig:
    crop_size: int = 512  # 768 also tested in the proposal
    pad_to_multiple: int = 16  # keeps tiling aligned for VLMs that need it


def _clamp_crop_box(cx: float, cy: float, crop: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Center a `crop`-sized window on (cx, cy), clamped to image bounds."""
    half = crop // 2
    x1 = round(cx - half)
    y1 = round(cy - half)
    x1 = max(0, min(x1, w - crop)) if w > crop else 0
    y1 = max(0, min(y1, h - crop)) if h > crop else 0
    x2 = min(w, x1 + crop)
    y2 = min(h, y1 + crop)
    return x1, y1, x2, y2


def crop_then_click(
    model: GUIModel,
    image: Image,
    instruction: str,
    *,
    cfg: CropConfig | None = None,
) -> ModelOutput:
    """Run the two-stage zoom-in policy. Returns a `ModelOutput` whose `.parsed.point`
    is in original-image pixel coordinates.
    """
    cfg = cfg or CropConfig()
    w, h = image.size

    coarse = model.predict(image, instruction)
    if coarse.parsed.point is None:
        # Stage 1 failed — return the failure verbatim so the caller can log it.
        return coarse

    cx, cy = coarse.parsed.point
    x1, y1, x2, y2 = _clamp_crop_box(cx, cy, cfg.crop_size, w, h)
    crop = image.crop((x1, y1, x2, y2))

    refined = model.predict(crop, instruction)
    if refined.parsed.point is None:
        # Stage 2 parse failed — fall back to the coarse click.
        return coarse

    rx, ry = refined.parsed.point
    final_point = (rx + x1, ry + y1)
    return ModelOutput(
        text=refined.text,
        parsed=ParsedAction(
            point=final_point,
            bbox=refined.parsed.bbox,
            raw=refined.parsed.raw,
            parser=f"crop+{refined.parsed.parser}",
        ),
        metadata={
            **refined.metadata,
            "crop_box": (x1, y1, x2, y2),
            "coarse_point": (cx, cy),
            "crop_size": cfg.crop_size,
        },
    )
