"""OSWorld-G loader (the grounding subset extracted from OSWorld scenes)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import warnings

from .screenspot import _try_row_to_sample
from .types import GroundingSample

# The canonical OSWorld-G mirror. `xlangai/OSWorld-G` (cited in the proposal)
# does not exist as of 2026-05-20; `MMInstruction/OSWorld-G` is the actual
# release with a parquet split that `load_dataset` can read directly.
OSWORLD_G_REPO = "MMInstruction/OSWorld-G"


def load_osworld_g(
    split: str = "test",
    *,
    repo_id: str = OSWORLD_G_REPO,
    streaming: bool = False,
    **kwargs: Any,
) -> Iterator[GroundingSample]:
    """Yield OSWorld-G samples (~564 grounding tasks from real OSWorld scenes)."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, streaming=streaming, **kwargs)
    skipped = 0
    for i, row in enumerate(ds):
        # MMInstruction/OSWorld-G names the click box `mimo_bbox` (x1,y1,x2,y2) /
        # `box_coordinates` (x,y,w,h) — neither is a key the shared converter
        # recognises, so promote one to `bbox` with the right format.
        norm = dict(row)
        if norm.get("mimo_bbox") is not None:
            norm["bbox"], fmt = norm["mimo_bbox"], "xyxy"
        elif norm.get("box_coordinates") is not None:
            norm["bbox"], fmt = norm["box_coordinates"], "xywh"
        else:
            fmt = None
        sample = _try_row_to_sample(norm, "osworld-g", i, bbox_format=fmt)
        if sample is None:
            skipped += 1
            continue
        yield sample
    if skipped:
        warnings.warn(
            f"Skipped {skipped} invalid OSWorld-G rows with missing image/bbox.",
            stacklevel=2,
        )
