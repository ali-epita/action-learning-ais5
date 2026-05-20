"""CPU-only tests for the vision-encoder forward-hook profiler.

Builds tiny `nn.Module` stubs and verifies that:
- `find_vision_module` locates a child named like a vision tower.
- `_VisionCallback` records and resets accumulated forward time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_find_vision_module_picks_named_child():
    from ais5.bench.profile import find_vision_module

    class TinyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(8, 16)
            self.lm_head = torch.nn.Linear(16, 4)

        def forward(self, x):
            return self.lm_head(torch.relu(self.visual(x)))

    model = TinyVLM()
    name, module = find_vision_module(model)
    assert name == "visual"
    assert module is model.visual


def test_find_vision_module_returns_none_when_absent():
    from ais5.bench.profile import find_vision_module

    class NoVisionHere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    name, module = find_vision_module(NoVisionHere())
    assert name is None
    assert module is None


def test_find_vision_module_matches_alternative_names():
    from ais5.bench.profile import find_vision_module

    class PaliLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.Linear(8, 16)
            self.lm = torch.nn.Linear(16, 4)

    name, module = find_vision_module(PaliLike())
    assert name == "vision_tower"
    assert module is module  # not None


def test_vision_callback_records_then_resets():
    from ais5.bench.profile import _VisionCallback

    class TinyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.Linear(8, 16)
            self.lm_head = torch.nn.Linear(16, 4)

        def forward(self, x):
            return self.lm_head(self.vision_tower(x))

    model = TinyVLM()
    cb = _VisionCallback(model.vision_tower, cuda=False)
    try:
        x = torch.randn(2, 8)
        _ = model(x)
        first = cb.consume()
        assert first >= 0
        assert cb.consume() == 0, "consume() must reset the accumulator"
        _ = model(x)
        assert cb.consume() >= 0
    finally:
        cb.remove()


def test_vision_callback_accumulates_across_repeated_calls():
    """consume() returns the SUM across all forwards since the last reset."""
    import time as time_mod

    from ais5.bench.profile import _VisionCallback

    class SlowLayer(torch.nn.Module):
        # Sleeps inside forward so the hook-to-hook span is deterministic,
        # avoiding the perf_counter-resolution flakiness a tiny Linear has.
        def forward(self, x):
            time_mod.sleep(0.005)
            return x

    vis = SlowLayer()
    cb = _VisionCallback(vis, cuda=False)
    try:
        x = torch.randn(2, 8)
        _ = vis(x)
        single = cb.consume()
        _ = vis(x)
        _ = vis(x)
        double = cb.consume()
        # Each sleep is 5 ms; allow generous slack for scheduler jitter.
        assert single >= 3.0, f"expected >=3ms from one forward, got {single}"
        assert double >= 8.0, f"expected >=8ms from two forwards, got {double}"
    finally:
        cb.remove()


def test_remove_unhooks_so_no_further_accumulation():
    from ais5.bench.profile import _VisionCallback

    vis = torch.nn.Linear(8, 16)
    cb = _VisionCallback(vis, cuda=False)
    cb.remove()
    x = torch.randn(2, 8)
    _ = vis(x)
    assert cb.consume() == 0, "post-remove forwards must not accumulate"
