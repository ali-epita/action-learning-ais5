"""OSWorld-G loader (the grounding subset extracted from OSWorld scenes)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .screenspot import _row_to_sample
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
    for i, row in enumerate(ds):
        yield _row_to_sample(row, "osworld-g", i)
