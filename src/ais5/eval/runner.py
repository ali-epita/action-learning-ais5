"""Generic eval loop: model × benchmark → list of `ClickResult`."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from ..data.types import GroundingSample
from .click import ClickResult, point_in_bbox

if TYPE_CHECKING:
    from ..models.base import GUIModel


@dataclass
class EvalRun:
    """Container for results plus the metadata needed to reproduce a run."""

    benchmark: str
    model_name: str
    results: list[ClickResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def avg_latency_ms(self) -> float | None:
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)


def evaluate_model(
    model: GUIModel,
    samples: Iterable[GroundingSample],
    *,
    benchmark: str | None = None,
    limit: int | None = None,
    progress: bool = True,
    on_predict: callable | None = None,  # type: ignore[type-arg]
) -> EvalRun:
    """Run `model.predict` over every sample and score against the gold bbox.

    Parameters
    ----------
    model:
        Anything implementing `GUIModel`.
    samples:
        Iterable of `GroundingSample`, e.g. from `load_benchmark(...)`.
    limit:
        Stop after this many samples (useful when validating a pipeline).
    on_predict:
        Optional callback `(sample, prediction, latency_ms) -> None`.
        Lets the benchmark module hook in extra timing without subclassing.
    """
    iterator: Iterator[GroundingSample] = iter(samples)
    if limit is not None:
        iterator = _truncate(iterator, limit)
    if progress:
        iterator = tqdm(iterator, desc=f"eval {model.name}", total=limit)

    bench_name = benchmark
    results: list[ClickResult] = []

    for sample in iterator:
        bench_name = bench_name or sample.benchmark
        t0 = time.perf_counter()
        out = model.predict(sample.image, sample.instruction)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if on_predict is not None:
            on_predict(sample, out, dt_ms)

        pred = out.parsed.point
        correct = bool(pred is not None and point_in_bbox(pred, sample.bbox))
        results.append(
            ClickResult(
                sample_id=sample.sample_id or "",
                pred=pred,
                bbox=sample.bbox,
                correct=correct,
                benchmark=sample.benchmark,
                target_type=sample.target_type,
                ui_type=sample.ui_type,
                target_relative_area=sample.target_relative_area,
                raw_response=out.text,
                latency_ms=dt_ms,
            )
        )

    return EvalRun(
        benchmark=bench_name or "",
        model_name=model.name,
        results=results,
        config={"limit": limit},
    )


def _truncate(it: Iterator[GroundingSample], n: int) -> Iterator[GroundingSample]:
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x
