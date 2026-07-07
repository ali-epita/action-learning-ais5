"""Coordinate mapping between the grounding image and the real screen.

The chain has three coordinate spaces:

    grounding image  (what the model sees; possibly downscaled)
        │  ÷ ground_scale
        ▼
    capture physical px  (mss grab; Retina = 2× logical)
        │  + monitor offset, then ÷ retina_scale
        ▼
    logical screen points  (what pyautogui clicks in)

``CoordinateMapper`` is pure and headless-testable — no Qt, no mss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def is_degenerate_point(point: tuple[float, float]) -> bool:
    """True for a grounding-image point at the very top-left corner — the
    model's non-answer for "not found", which must never be clicked (it also
    trips pyautogui's corner FAILSAFE)."""
    x, y = point
    return x <= 2 and y <= 2


def validate_logical_point(
    lx: float, ly: float, logical_size: tuple[int, int], margin: float = 8.0
) -> tuple[float, float] | None:
    """Vet a mapped click point against the logical screen bounds before it is
    allowed anywhere near a real click. A small overshoot (parser rounding, an
    edge answer like <click>1000,1000</click>) is clamped back inside; anything
    farther out — or non-finite — is a model failure and returns None. The
    clamp stays 1-2 px off the exact corners so a legitimate edge click cannot
    land on a pyautogui FAILSAFE corner."""
    lw, lh = logical_size
    if lw <= 0 or lh <= 0 or not (math.isfinite(lx) and math.isfinite(ly)):
        return None
    if not (-margin <= lx <= lw + margin and -margin <= ly <= lh + margin):
        return None
    return (min(max(lx, 1.0), lw - 2.0), min(max(ly, 1.0), lh - 2.0))


def compute_ground_scale(cap_w: int, cap_h: int, max_side: int | None) -> float:
    """Factor to multiply the capture by so its long side <= ``max_side``.
    Returns 1.0 (no upscaling) when the capture already fits or max_side is falsy."""
    if not max_side:
        return 1.0
    longest = max(cap_w, cap_h)
    return min(1.0, max_side / longest) if longest > 0 else 1.0


@dataclass(frozen=True)
class CoordinateMapper:
    capture_size: tuple[int, int]  # physical px of the captured region
    logical_size: tuple[int, int]  # pyautogui logical size of that region
    ground_scale: float  # grounding_image_size = capture_size * ground_scale
    offset_physical: tuple[int, int] = (0, 0)  # capture origin in virtual-desktop physical px

    @property
    def retina_scale(self) -> float:
        lw = self.logical_size[0] or self.capture_size[0]
        return self.capture_size[0] / lw if lw else 1.0

    @property
    def grounding_size(self) -> tuple[int, int]:
        return (round(self.capture_size[0] * self.ground_scale),
                round(self.capture_size[1] * self.ground_scale))

    def ground_to_logical(self, gx: float, gy: float) -> tuple[float, float]:
        """Grounding-image pixel → logical screen point (for pyautogui)."""
        px = gx / self.ground_scale + self.offset_physical[0]
        py = gy / self.ground_scale + self.offset_physical[1]
        s = self.retina_scale
        return (px / s, py / s)

    def ground_to_physical(self, gx: float, gy: float) -> tuple[float, float]:
        """Grounding-image pixel → virtual-desktop physical px (for an overlay
        that paints in device pixels)."""
        return (gx / self.ground_scale + self.offset_physical[0],
                gy / self.ground_scale + self.offset_physical[1])

    def physical_to_logical(self, px: float, py: float) -> tuple[float, float]:
        s = self.retina_scale
        return (px / s, py / s)
