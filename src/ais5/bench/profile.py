"""Drives the full Task 3 benchmark: model × quantization × benchmark grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..eval import evaluate_model
from ..eval.runner import EvalRun
from ..utils.logging import get_logger
from .latency import LatencyStats
from .memory import MemoryStats, measure_peak_memory

if TYPE_CHECKING:
    from ..models.base import GUIModel

log = get_logger(__name__)


@dataclass
class BenchResult:
    model: str
    quant: str
    benchmark: str
    accuracy: float
    n_samples: int
    latency: LatencyStats
    memory: MemoryStats
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "quant": self.quant,
            "benchmark": self.benchmark,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            **{f"latency_{k}": v for k, v in self.latency.to_dict().items()},
            **self.memory.to_dict(),
            **self.extra,
        }


def run_full_benchmark(
    model: GUIModel,
    samples: list,
    *,
    benchmark: str,
    quant_label: str = "fp16",
    limit: int | None = None,
) -> BenchResult:
    """Run accuracy + latency + memory measurement for one (model, quant, bench) cell."""
    log.info("Benchmarking %s on %s [%s]", model.name, benchmark, quant_label)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    latencies = LatencyStats()

    def _record(_sample: Any, _out: Any, dt_ms: float) -> None:
        latencies.add(dt_ms)

    run: EvalRun = evaluate_model(
        model,
        samples,
        benchmark=benchmark,
        limit=limit,
        on_predict=_record,
    )

    mem = measure_peak_memory()
    return BenchResult(
        model=model.name,
        quant=quant_label,
        benchmark=benchmark,
        accuracy=run.accuracy,
        n_samples=len(run.results),
        latency=latencies,
        memory=mem,
    )
