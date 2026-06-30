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

from dataclasses import dataclass


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
