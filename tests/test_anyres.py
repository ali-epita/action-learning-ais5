"""Tests for resolution scaling and AnyRes tiling."""

from __future__ import annotations

from PIL import Image

from ais5.tile.anyres import AnyResConfig, scale_image, tile_anyres


def test_scale_image_identity():
    img = Image.new("RGB", (100, 50))
    assert scale_image(img, 1.0).size == (100, 50)


def test_scale_image_half_and_double():
    img = Image.new("RGB", (100, 50))
    assert scale_image(img, 0.5).size == (50, 25)
    assert scale_image(img, 2.0).size == (200, 100)


def test_tile_anyres_includes_thumbnail():
    img = Image.new("RGB", (1024, 1024))
    tiles = tile_anyres(img, AnyResConfig(tile_size=512, max_tiles=4, keep_thumbnail=True))
    assert len(tiles) >= 2
    # First entry is the thumbnail, smaller than tile_size.
    thumb, offset = tiles[0]
    assert offset == (0, 0)
    assert max(thumb.size) <= 384


def test_tile_anyres_no_thumbnail_returns_only_tiles():
    img = Image.new("RGB", (1024, 1024))
    tiles = tile_anyres(img, AnyResConfig(tile_size=512, max_tiles=4, keep_thumbnail=False))
    # 1024×1024 / 512 = 4 full tiles
    assert len(tiles) == 4
    sizes = [t[0].size for t in tiles]
    assert all(s == (512, 512) for s in sizes)


def test_tile_anyres_respects_max_tiles():
    img = Image.new("RGB", (4096, 4096))
    tiles = tile_anyres(img, AnyResConfig(tile_size=512, max_tiles=4, keep_thumbnail=False))
    assert len(tiles) == 4
