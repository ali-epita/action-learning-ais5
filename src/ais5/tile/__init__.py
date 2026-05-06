"""Resolution scaling + crop-then-click (Task 2)."""

from .anyres import AnyResConfig, scale_image, tile_anyres
from .crop_then_click import CropConfig, crop_then_click

__all__ = [
    "AnyResConfig",
    "CropConfig",
    "crop_then_click",
    "scale_image",
    "tile_anyres",
]
