"""Prove the lazy eval loop bounds RAM versus list() materialization.

Runs entirely on CPU with a mock model and synthetic high-res images, so it
needs no GPU or network. Reports peak RSS for the old list(load_benchmark(...))
pattern against the new lazy iter_benchmark pattern.

    uv run python scripts/smoke_ram.py
"""

from __future__ import annotations

import gc

import psutil
from PIL import Image as PILImage

from ais5.data import registry
from ais5.data.types import GroundingSample
from ais5.eval import evaluate_model
from ais5.models.base import GUIModel, ModelOutput
from ais5.prompt.action import parse_click

N = 60
SIZE = (3840, 2160)  # a 4K screenshot decodes to ~24 MB RGB
_PROC = psutil.Process()


def rss_gb() -> float:
    return _PROC.memory_info().rss / 1024**3


class MockModel(GUIModel):
    name = "mock"
    family = "generalist"
    param_count_b = 0.0

    def predict(self, image, instruction, **_kw):
        return ModelOutput(text="(1, 1)", parsed=parse_click("(1, 1)"))


def _fake_loader(**_kw):
    for i in range(N):
        img = PILImage.new("RGB", SIZE, (i % 255, 1, 1))
        yield GroundingSample(
            image=img,
            instruction="click",
            bbox=(0.0, 0.0, 2.0, 2.0),
            image_size=SIZE,
            benchmark="fake",
            sample_id=str(i),
        )


def main() -> None:
    registry._BENCHMARKS["fake"] = _fake_loader
    model = MockModel()
    base = rss_gb()

    # Lazy: peak sampled on each predict; only one image alive at a time.
    peak_lazy = base

    def track(_s, _o, _ms):
        nonlocal peak_lazy
        peak_lazy = max(peak_lazy, rss_gb())

    evaluate_model(model, registry.iter_benchmark("fake"), benchmark="fake",
                   progress=False, on_predict=track)
    gc.collect()

    # Old pattern: the whole benchmark is decoded into a list first.
    base2 = rss_gb()
    samples = list(registry.load_benchmark("fake"))
    peak_list = rss_gb()
    evaluate_model(model, samples, benchmark="fake", progress=False)
    del samples
    gc.collect()

    full = N * SIZE[0] * SIZE[1] * 3 / 1024**3
    print(f"images: {N} x {SIZE[0]}x{SIZE[1]} (~{full:.1f} GB if all decoded)")
    print(f"lazy  iter_benchmark: +{peak_lazy - base:.2f} GB over baseline")
    print(f"old   list()        : +{peak_list - base2:.2f} GB over baseline")


if __name__ == "__main__":
    main()
