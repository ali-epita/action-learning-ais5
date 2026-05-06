"""Per-step latency probing for Task 3."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..models.base import GUIModel


@dataclass
class LatencyStats:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples_ms) if self.samples_ms else 0.0

    @property
    def p95(self) -> float:
        if not self.samples_ms:
            return 0.0
        sorted_ms = sorted(self.samples_ms)
        idx = max(0, round(0.95 * len(sorted_ms)) - 1)
        return sorted_ms[idx]

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.samples_ms) if self.n > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_ms": self.mean,
            "median_ms": self.median,
            "p95_ms": self.p95,
            "stdev_ms": self.stdev,
        }


def measure_latency(
    model: GUIModel,
    image: Image,
    instruction: str,
    *,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> LatencyStats:
    """Time `n_runs` calls to `model.predict` after `n_warmup` discarded calls.

    CUDA timings are synchronized to avoid measuring async kernel queueing.
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False

    for _ in range(n_warmup):
        model.predict(image, instruction)
    if cuda:
        import torch

        torch.cuda.synchronize()

    stats = LatencyStats()
    for _ in range(n_runs):
        if cuda:
            import torch

            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(image, instruction)
        if cuda:
            import torch

            torch.cuda.synchronize()
        stats.add((time.perf_counter() - t0) * 1000.0)
    return stats
