"""Peak VRAM (and RSS) measurement for Task 3.

The proposal pins an 8 GB on-device budget. Free Colab/Kaggle GPUs are 16 GB,
so we enforce the budget as a software cap and *measure* peak usage rather
than depend on physically smaller hardware. `vram_budget_check` returns a
boolean so the benchmark loop can mark a config "in-budget" or "over".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryStats:
    peak_vram_gb: float
    peak_rss_gb: float

    def to_dict(self) -> dict[str, float]:
        return {"peak_vram_gb": self.peak_vram_gb, "peak_rss_gb": self.peak_rss_gb}


def measure_peak_memory(*, reset: bool = True) -> MemoryStats:
    """Snapshot peak GPU + CPU memory since the last reset."""
    peak_vram = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            if reset:
                torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    peak_rss = 0.0
    try:
        import psutil

        peak_rss = psutil.Process().memory_info().rss / (1024**3)
    except ImportError:
        pass

    return MemoryStats(peak_vram_gb=peak_vram, peak_rss_gb=peak_rss)


def vram_budget_check(stats: MemoryStats, *, budget_gb: float = 8.0) -> bool:
    return stats.peak_vram_gb <= budget_gb
