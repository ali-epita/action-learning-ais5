"""iter_benchmark streams lazily and holds one decoded image at a time."""

from __future__ import annotations

import gc

from ais5.data import registry


def test_iter_benchmark_is_lazy(monkeypatch):
    produced: list[int] = []

    def fake_loader(**_kw):
        for i in range(100):
            produced.append(i)
            yield i

    monkeypatch.setitem(registry._BENCHMARKS, "fake", fake_loader)

    gen = registry.iter_benchmark("fake", limit=3)
    assert produced == []  # nothing pulled until iterated

    out = list(gen)
    assert out == [0, 1, 2]
    assert produced == [0, 1, 2]  # islice stopped the source, no over-pull


def test_iter_benchmark_no_limit(monkeypatch):
    monkeypatch.setitem(registry._BENCHMARKS, "fake", lambda **_k: iter(range(5)))
    assert list(registry.iter_benchmark("fake")) == [0, 1, 2, 3, 4]


def test_iter_benchmark_one_image_in_ram(monkeypatch):
    live = {"count": 0, "max": 0}

    class FakeImg:
        def __init__(self):
            live["count"] += 1
            live["max"] = max(live["max"], live["count"])

        def __del__(self):
            live["count"] -= 1

    def fake_loader(**_kw):
        for _ in range(10):
            yield FakeImg()

    monkeypatch.setitem(registry._BENCHMARKS, "fake", fake_loader)

    for sample in registry.iter_benchmark("fake"):
        del sample
        gc.collect()

    assert live["max"] == 1
