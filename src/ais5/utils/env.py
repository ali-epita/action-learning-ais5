"""Colab/Kaggle runtime helpers: cache redirection and GPU teardown."""

from __future__ import annotations

import os
from typing import Any

from .io import ensure_dir
from .logging import get_logger

log = get_logger(__name__)

_CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE")


def _on_colab() -> bool:
    import sys

    return "google.colab" in sys.modules


def configure_caches(root: str | None = None) -> dict[str, str]:
    """Point HF and torch caches at `root`. Call before importing transformers.

    Only sets vars that are currently unset, so an explicit HF_HOME wins —
    including for the derived sub-caches: when HF_HOME is preset (e.g. pinned
    to a RunPod volume), HF_HUB_CACHE/HF_DATASETS_CACHE default under IT, not
    under the generic default root (huggingface_hub gives HF_HUB_CACHE
    precedence, so deriving it from the wrong root would silently re-download
    models outside the pinned volume).
    Defaults to /content/hf_cache on Colab, ~/.cache elsewhere.
    """
    if root is None:
        root = (
            os.environ.get("HF_HOME")
            or ("/content/hf_cache" if _on_colab() else os.path.expanduser("~/.cache"))
        )
    root = str(ensure_dir(root))

    layout = {
        "HF_HOME": root,
        "HF_HUB_CACHE": os.path.join(root, "hub"),
        "HF_DATASETS_CACHE": os.path.join(root, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(root, "transformers"),
        "TORCH_HOME": os.path.join(root, "torch"),
    }
    resolved: dict[str, str] = {}
    for var, path in layout.items():
        if os.environ.get(var):
            resolved[var] = os.environ[var]
            continue
        ensure_dir(path)
        os.environ[var] = path
        resolved[var] = path
    log.info("HF/torch caches under %s", root)
    return resolved


def free_model(*objs: Any) -> None:
    """Release models and reclaim GPU/host memory between grid cells.

    Drops references, runs gc.collect() so bitsandbytes/PEFT graphs are
    collected, then empties the CUDA cache. The caller should also del its own
    variable.
    """
    import gc

    del objs
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def purge_hf_cache(repo_id: str) -> int:
    """Delete all cached revisions of one HF repo and return bytes freed."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return 0

    cache = scan_cache_dir()
    revisions = [
        rev.commit_hash
        for repo in cache.repos
        if repo.repo_id == repo_id
        for rev in repo.revisions
    ]
    if not revisions:
        return 0
    strategy = cache.delete_revisions(*revisions)
    freed = strategy.expected_freed_size
    strategy.execute()
    log.info("Purged %s from HF cache (%.2f GB)", repo_id, freed / (1024**3))
    return freed
