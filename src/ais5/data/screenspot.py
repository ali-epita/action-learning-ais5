"""ScreenSpot-V2 and ScreenSpot-Pro loaders.

Both benchmarks ship on the Hugging Face Hub. Schemas are slightly different
between the two (and between mirrors), so `_row_to_sample` is defensive and
uses fallback keys.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .types import GroundingSample

# Default mirrors. Override via the env vars or by passing `repo_id` directly.
SCREENSPOT_V2_REPO = "likaixin/ScreenSpot-v2-variants"
SCREENSPOT_PRO_REPO = "likaixin/ScreenSpot-Pro"


def _coerce_bbox(raw: Any) -> tuple[float, float, float, float]:
    """Accept (x1, y1, x2, y2) or (x, y, w, h) lists/tuples; return (x1, y1, x2, y2)."""
    if raw is None:
        raise ValueError("missing bbox")
    coords = [float(v) for v in raw]
    if len(coords) != 4:
        raise ValueError(f"expected 4 bbox values, got {len(coords)}: {raw}")
    x1, y1, a, b = coords
    if a < x1 or b < y1:
        return (x1, y1, x1 + a, y1 + b)
    return (x1, y1, a, b)


def _first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _row_to_sample(row: dict[str, Any], benchmark: str, idx: int) -> GroundingSample:
    image = row["image"]
    bbox = _coerce_bbox(_first_present(row, "bbox", "bounding_box", "box"))
    instruction = _first_present(row, "instruction", "query", "task", default="")
    target_type = _first_present(row, "data_type", "element_type", "target_type")
    ui_type = _first_present(row, "data_source", "platform", "application", "group")
    sample_id = _first_present(row, "id", "image_id", default=str(idx))

    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "image",
            "bbox",
            "bounding_box",
            "box",
            "instruction",
            "query",
            "task",
        }
    }

    return GroundingSample(
        image=image,
        instruction=str(instruction),
        bbox=bbox,
        image_size=image.size,
        benchmark=benchmark,
        target_type=str(target_type) if target_type else None,
        ui_type=str(ui_type) if ui_type else None,
        sample_id=str(sample_id),
        extra=extras,
    )


def _load_iter(
    repo_id: str,
    benchmark: str,
    split: str,
    *,
    streaming: bool = False,
    **load_kwargs: Any,
) -> Iterator[GroundingSample]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, streaming=streaming, **load_kwargs)
    for i, row in enumerate(ds):
        yield _row_to_sample(row, benchmark, i)


def load_screenspot_v2(
    split: str = "test",
    *,
    repo_id: str = SCREENSPOT_V2_REPO,
    streaming: bool = False,
    **kwargs: Any,
) -> Iterator[GroundingSample]:
    """Yield ScreenSpot-V2 samples (1,272 in the test split)."""
    yield from _load_iter(repo_id, "screenspot-v2", split, streaming=streaming, **kwargs)


def load_screenspot_pro(
    split: str = "test",
    *,
    repo_id: str = SCREENSPOT_PRO_REPO,
    streaming: bool = False,
    **kwargs: Any,
) -> Iterator[GroundingSample]:
    """Yield ScreenSpot-Pro samples (~1,581 high-resolution professional UIs)."""
    yield from _load_iter(repo_id, "screenspot-pro", split, streaming=streaming, **kwargs)
