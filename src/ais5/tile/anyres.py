"""AnyRes-style tiling + a simple resolution-scaling helper.

The proposal's resolution sweep (0.5×, 1×, 2× native) only needs `scale_image`.
The full AnyRes tiling is exposed for Task 2's crop-then-click and for any
future pipeline that wants tile budgets matched across models.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image as PILImage
from PIL.Image import Image


@dataclass
class AnyResConfig:
    tile_size: int = 512
    max_tiles: int = 12
    keep_thumbnail: bool = True
    thumbnail_size: int = 384


def scale_image(image: Image, factor: float, *, resample: int = PILImage.LANCZOS) -> Image:
    """Resize `image` by `factor` (e.g. 0.5, 1.0, 2.0)."""
    if factor == 1.0:
        return image
    w, h = image.size
    new_size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return image.resize(new_size, resample=resample)


def tile_anyres(
    image: Image, cfg: AnyResConfig | None = None
) -> list[tuple[Image, tuple[int, int]]]:
    """Split `image` into non-overlapping tiles of `cfg.tile_size`.

    Returns a list of `(tile, (x_offset, y_offset))` pairs. If
    `cfg.keep_thumbnail` is True, the first entry is a downscaled overview.
    """
    cfg = cfg or AnyResConfig()
    tiles: list[tuple[Image, tuple[int, int]]] = []

    if cfg.keep_thumbnail:
        thumb = image.copy()
        thumb.thumbnail((cfg.thumbnail_size, cfg.thumbnail_size))
        tiles.append((thumb, (0, 0)))

    w, h = image.size
    for y in range(0, h, cfg.tile_size):
        for x in range(0, w, cfg.tile_size):
            box = (x, y, min(x + cfg.tile_size, w), min(y + cfg.tile_size, h))
            tiles.append((image.crop(box), (x, y)))
            if len(tiles) >= cfg.max_tiles + (1 if cfg.keep_thumbnail else 0):
                return tiles
    return tiles
