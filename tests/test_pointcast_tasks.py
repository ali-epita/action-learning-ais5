"""Headless tests for PointCast task mode: intent matching + the runner loop.

No Qt, no model, no screen: the runner is driven with stub engine/clicker/capture
and a callback recorder, so the whole multi-step loop is exercised deterministically.
"""

from __future__ import annotations

import types

from ais5.pointcast.config import PointCastConfig
from ais5.pointcast.engine import GroundResult
from ais5.pointcast.task import TaskCallbacks, TaskRunner, match_recipe
from ais5.pointcast.task.recipes import Recipe, Step


# ── intent matching ──────────────────────────────────────────────────────────
def test_intent_matches_storage_phrases():
    assert match_recipe("how much free storage do i have").name == "check storage"
    assert match_recipe("what's my disk space").name == "check storage"


def test_intent_ignores_ordinary_targets():
    assert match_recipe("click the search box") is None
    assert match_recipe("the send button") is None
    assert match_recipe("") is None


# ── runner loop ──────────────────────────────────────────────────────────────
class StubEngine:
    def __init__(self, point=(100.0, 100.0), accepted=True, answer="You have 120 GB available."):
        self._pt, self._acc, self._ans = point, accepted, answer
        self.ground_calls: list[str] = []
        self.ask_calls: list[str] = []

    def ground(self, image, target):
        self.ground_calls.append(target)
        return GroundResult(point=self._pt, accepted=self._acc, badge="locked", candidates=[self._pt])

    def ask(self, image, prompt, *, max_tokens=128):
        self.ask_calls.append(prompt)
        return self._ans


class RecClicker:
    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append((x, y))


class _IdMapper:
    def ground_to_logical(self, x, y):
        return (x, y)


def _capture():
    return types.SimpleNamespace(image=None, mapper=_IdMapper())


def _recorder():
    r = {"status": [], "say": [], "point": [], "answer": [], "failed": [], "clear": 0, "done": 0}
    cb = TaskCallbacks(
        status=r["status"].append,
        say=r["say"].append,
        point=lambda x, y: r["point"].append((x, y)),
        clear=lambda: r.__setitem__("clear", r["clear"] + 1),
        answer=r["answer"].append,
        failed=r["failed"].append,
        done=lambda: r.__setitem__("done", r["done"] + 1),
    )
    return r, cb


RECIPE = Recipe(name="t", utterances=("t",),
                steps=(Step("settings", settle_ms=0), Step("storage", settle_ms=0)),
                question="how much free space?")


def test_runner_navigates_clicks_and_reads_answer():
    cfg = PointCastConfig(task_preview_ms=0)  # real clicks, no waits
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert eng.ground_calls == ["settings", "storage"]
    assert clk.calls == [(100.0, 100.0), (100.0, 100.0)]
    assert r["answer"] == ["You have 120 GB available."]
    assert eng.ask_calls == ["how much free space?"]
    assert r["done"] == 1 and not r["failed"]


def test_runner_stops_on_uncertain_step():
    cfg = PointCastConfig(task_preview_ms=0)
    eng, clk = StubEngine(accepted=False), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []  # never clicked
    assert r["failed"] and r["done"] == 1
    assert not r["answer"]


def test_runner_dry_run_does_not_click():
    cfg = PointCastConfig(task_preview_ms=0, dry_run=True)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []  # dry-run: grounded + read answer, but no real clicks
    assert eng.ground_calls == ["settings", "storage"]
    assert r["answer"] == ["You have 120 GB available."]


def test_runner_deep_link_skips_clicking():
    cfg = PointCastConfig(task_preview_ms=0, dry_run=True, use_deep_links=True)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    recipe = Recipe(name="d", utterances=("d",), steps=(Step("x"),),
                    deep_link="x-apple.systempreferences:com.apple.settings.Storage",
                    question="q?")
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert eng.ground_calls == []  # deep-link path: no per-step grounding
    assert r["answer"] == ["You have 120 GB available."]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} task tests passed")
