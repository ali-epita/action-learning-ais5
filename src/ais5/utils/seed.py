"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Imports of numpy/torch are lazy so callers without them installed (e.g. CI
    on a machine without CUDA) don't pay for the import.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
