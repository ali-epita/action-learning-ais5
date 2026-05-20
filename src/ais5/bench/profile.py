"""Drives the full Task 3 benchmark: model x quantization x benchmark grid.

Also provides per-component profiling: forward hooks on the vision tower
attribute per-step latency between the encoder and "everything else"
(LLM decode + projector). That breakdown is Task 3's deeper-analysis
component per the modeling-task PDF.
"""

from __future__ import annotations

import statistics
import time
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

# Attribute names that commonly host the vision encoder across HF VLMs.
# First match by `named_modules()` order wins — fine for VLMs with exactly
# one vision tower (every model in our grid).
_VISION_MODULE_NAMES = (
    "visual",        # Qwen2-VL / Qwen2.5-VL family — OS-Atlas, ShowUI, Qwen
    "vision_tower",  # PaliGemma (SigLIP), LLaVA-style
    "vision_model",  # InternVL, CLIP-style
    "vit",
    "image_encoder",
)


@dataclass
class ComponentTimings:
    """Per-call breakdown of vision-encode time vs. the rest of the forward pass."""

    visual_encode_ms_per_call: list[float] = field(default_factory=list)
    total_ms_per_call: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.total_ms_per_call)

    @property
    def visual_encode_mean_ms(self) -> float:
        if not self.visual_encode_ms_per_call:
            return 0.0
        return statistics.fmean(self.visual_encode_ms_per_call)

    @property
    def total_mean_ms(self) -> float:
        if not self.total_ms_per_call:
            return 0.0
        return statistics.fmean(self.total_ms_per_call)

    @property
    def llm_decode_mean_ms(self) -> float:
        return max(0.0, self.total_mean_ms - self.visual_encode_mean_ms)

    @property
    def visual_encode_share(self) -> float:
        if self.total_mean_ms <= 0:
            return 0.0
        return self.visual_encode_mean_ms / self.total_mean_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "visual_encode_mean_ms": self.visual_encode_mean_ms,
            "llm_decode_mean_ms": self.llm_decode_mean_ms,
            "total_mean_ms": self.total_mean_ms,
            "visual_encode_share": self.visual_encode_share,
        }


def find_vision_module(root: Any) -> tuple[str | None, Any]:
    """Walk `root.named_modules()` for an attribute named like a vision tower.

    Returns `(qualified_name, module)` or `(None, None)` if nothing matches.
    """
    for name, module in root.named_modules():
        bare = name.rsplit(".", 1)[-1]
        if bare in _VISION_MODULE_NAMES:
            return name, module
    return None, None


class _VisionCallback:
    """Forward-hook helper that accumulates vision-encoder time across one+ calls.

    Build with the vision module, then call `consume()` after each predict to
    read and reset the accumulator. Accumulates rather than overwriting so
    multi-image predict calls still report the total encode time.
    """

    def __init__(self, vision_module: Any, *, cuda: bool) -> None:
        self.cuda = cuda
        self._t0: float | None = None
        self._accum_ms: float = 0.0
        self._handles = [
            vision_module.register_forward_pre_hook(self._pre),
            vision_module.register_forward_hook(self._post),
        ]

    def _pre(self, _module: Any, _inputs: Any) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def _post(self, _module: Any, _inputs: Any, _output: Any) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()
        if self._t0 is not None:
            self._accum_ms += (time.perf_counter() - self._t0) * 1000.0
            self._t0 = None

    def consume(self) -> float:
        ms = self._accum_ms
        self._accum_ms = 0.0
        return ms

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@dataclass
class BenchResult:
    model: str
    quant: str
    benchmark: str
    accuracy: float
    n_samples: int
    latency: LatencyStats
    memory: MemoryStats
    components: ComponentTimings | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "quant": self.quant,
            "benchmark": self.benchmark,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            **{f"latency_{k}": v for k, v in self.latency.to_dict().items()},
            **self.memory.to_dict(),
        }
        if self.components is not None:
            out.update({f"components_{k}": v for k, v in self.components.to_dict().items()})
        out.update(self.extra)
        return out


def run_full_benchmark(
    model: GUIModel,
    samples: list,
    *,
    benchmark: str,
    quant_label: str = "fp16",
    limit: int | None = None,
    measure_components: bool = False,
) -> BenchResult:
    """Run accuracy + latency + memory for one (model, quant, bench) cell.

    With `measure_components=True`, attach a forward hook to the model's vision
    tower so we can attribute per-step time between visual encode and the rest
    (LLM decode + projector). Silently no-op for models where no recognised
    vision attribute is found — logs a warning in that case.
    """
    log.info("Benchmarking %s on %s [%s]", model.name, benchmark, quant_label)

    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda = False

    latencies = LatencyStats()
    components: ComponentTimings | None = None
    cb: _VisionCallback | None = None

    if measure_components and getattr(model, "model", None) is not None:
        name, vision_module = find_vision_module(model.model)
        if vision_module is not None:
            cb = _VisionCallback(vision_module, cuda=cuda)
            components = ComponentTimings()
            log.info("Component profiling hooked at %s.%s",
                     type(model.model).__name__, name)
        else:
            log.warning("No vision module found on %s; skipping component profiling",
                        model.name)

    def _record(_sample: Any, _out: Any, dt_ms: float) -> None:
        latencies.add(dt_ms)
        if components is not None and cb is not None:
            components.visual_encode_ms_per_call.append(cb.consume())
            components.total_ms_per_call.append(dt_ms)

    try:
        run: EvalRun = evaluate_model(
            model,
            samples,
            benchmark=benchmark,
            limit=limit,
            on_predict=_record,
        )
    finally:
        if cb is not None:
            cb.remove()

    mem = measure_peak_memory()
    return BenchResult(
        model=model.name,
        quant=quant_label,
        benchmark=benchmark,
        accuracy=run.accuracy,
        n_samples=len(run.results),
        latency=latencies,
        memory=mem,
        components=components,
    )
