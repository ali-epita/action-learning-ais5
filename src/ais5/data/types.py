"""Common dataclass shared across all benchmarks.

Each benchmark loader normalizes its native schema into a `GroundingSample`,
so downstream code (eval, runners, notebooks) is benchmark-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Image

Bbox = tuple[float, float, float, float]


@dataclass
class GroundingSample:
    image: Image
    instruction: str
    bbox: Bbox  # (x1, y1, x2, y2) in pixels relative to `image_size`
    image_size: tuple[int, int]  # (width, height)
    benchmark: str
    split: str = "test"
    target_type: str | None = None  # "icon" | "text" | …
    ui_type: str | None = None  # "web" | "desktop" | "mobile" | …
    sample_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def target_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def target_relative_area(self) -> float:
        w, h = self.image_size
        denom = max(1, w * h)
        return self.target_area / denom

    def __repr__(self) -> str:
        return (
            f"GroundingSample(id={self.sample_id!r}, benchmark={self.benchmark!r}, "
            f"image_size={self.image_size}, target_type={self.target_type!r})"
        )
