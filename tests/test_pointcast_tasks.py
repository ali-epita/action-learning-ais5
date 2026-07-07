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
    assert match_recipe("check my storage").name == "check storage"
    assert match_recipe("check storage").name == "check storage"


def test_intent_matches_battery_and_wifi():
    assert match_recipe("what's my battery percentage").name == "check battery"
    assert match_recipe("which wifi am i on").name == "check wifi"
    assert match_recipe("check wifi").name == "check wifi"


def test_kaggle_is_not_a_builtin_recipe():
    # The kaggle web task is recorded per-user (its targets are machine- and
    # layout-specific), not shipped as a built-in.
    assert match_recipe("check my kaggle week grade") is None
    assert match_recipe("what's my kaggle grade") is None


def test_intent_ignores_ordinary_targets():
    assert match_recipe("click the search box") is None
    assert match_recipe("the send button") is None
    assert match_recipe("open the file menu") is None
    assert match_recipe("") is None


def test_intent_single_word_triggers_do_not_hijack():
    # "wi-fi" is a built-in utterance but single-word triggers are ignored, so
    # pointing at a wifi-related UI element still single-clicks.
    assert match_recipe("the wi-fi settings icon") is None
    assert match_recipe("the wi-fi icon in the menu bar") is None


def test_intent_matches_on_word_boundaries():
    assert match_recipe("open storage spaces") is None  # not "storage space"
    assert match_recipe("what's my storage space") .name == "check storage"


def test_intent_click_prefix_is_a_literal_escape_hatch():
    # Prefixing with "click " always grounds literally, never runs a recipe.
    assert match_recipe("click check storage") is None
    assert match_recipe("click check battery") is None


# ── runner loop ──────────────────────────────────────────────────────────────
class StubEngine:
    def __init__(self, point=(100.0, 100.0), accepted=True, answer="You have 120 GB available."):
        self._pt, self._acc, self._ans = point, accepted, answer
        self.ground_calls: list[str] = []
        self.ask_calls: list[str] = []

    def ground(self, image, target, **kwargs):
        self.ground_calls.append(target)
        self.last_ground_kwargs = kwargs
        return GroundResult(point=self._pt, accepted=self._acc, badge="locked", candidates=[self._pt])

    def ask(self, image, prompt, *, max_tokens=128):
        self.ask_calls.append(prompt)
        if isinstance(self._ans, list):  # scripted per-call answers
            return self._ans[min(len(self.ask_calls) - 1, len(self._ans) - 1)]
        return self._ans


class RecClicker:
    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append((x, y))


class _IdMapper:
    logical_size = (1000, 1000)

    def ground_to_logical(self, x, y):
        return (x, y)


def _capture():
    return types.SimpleNamespace(image=None, mapper=_IdMapper())


def _recorder():
    r = {"status": [], "say": [], "point": [], "answer": [], "failed": [], "clear": 0, "hide": 0, "done": 0}
    cb = TaskCallbacks(
        status=r["status"].append,
        say=r["say"].append,
        point=lambda x, y: r["point"].append((x, y)),
        clear=lambda: r.__setitem__("clear", r["clear"] + 1),
        hide=lambda: r.__setitem__("hide", r["hide"] + 1),
        answer=r["answer"].append,
        failed=r["failed"].append,
        done=lambda: r.__setitem__("done", r["done"] + 1),
    )
    return r, cb


RECIPE = Recipe(name="t", utterances=("t",),
                steps=(Step("settings", settle_ms=0), Step("storage", settle_ms=0)),
                question="how much free space?")


def test_runner_navigates_clicks_and_reads_answer():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)  # real clicks, no waits
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert eng.ground_calls == ["settings", "storage"]
    assert clk.calls == [(100.0, 100.0), (100.0, 100.0)]
    assert r["answer"] == ["You have 120 GB available."]
    assert eng.ask_calls == ["how much free space?"]
    assert r["hide"] == 3  # overlays hidden before each of the 2 steps + the read-back
    assert r["done"] == 1 and not r["failed"]


def test_runner_stops_on_uncertain_step():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(accepted=False), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []  # never clicked
    assert r["failed"] and r["done"] == 1
    assert not r["answer"]


def test_runner_dry_run_does_not_click():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, dry_run=True)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []  # dry-run: grounded + read answer, but no real clicks
    assert eng.ground_calls == ["settings", "storage"]
    assert r["answer"] == ["You have 120 GB available."]


def test_runner_abort_during_preview_does_not_click():
    """The preview window is the user's veto: aborting there must prevent the click."""
    cfg = PointCastConfig(task_preview_ms=50, task_capture_hide_ms=0)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    runner = TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None)
    cb.point = lambda x, y: runner.abort()  # user hits the hotkey while the crosshair previews
    runner.run(RECIPE)
    assert clk.calls == []  # aborted before dispatch
    assert "Cancelled" in r["status"]
    assert r["done"] == 1


def test_runner_hide_gets_ack_event_and_waits_for_it():
    """_grab hands the UI thread an Event and proceeds once it is set."""
    import threading

    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    acks: list[object] = []

    def hide(ev=None):
        acks.append(ev)
        if ev is not None:
            ev.set()  # UI thread confirms the overlays are gone

    cb.hide = hide
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert len(acks) == 3 and all(isinstance(a, threading.Event) for a in acks)
    assert clk.calls  # and the task still ran to completion


def test_runner_waits_for_focus_settle_then_clicks():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, activate_settle_ms=0)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    focused: list[tuple] = []

    def focus(x, y, skip_pid=None):
        focused.append((x, y, skip_pid))
        return True  # a background app was raised

    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=focus).run(RECIPE)
    assert len(focused) == 2  # once per step
    assert clk.calls == [(100.0, 100.0), (100.0, 100.0)]


def test_runner_degenerate_corner_click_is_not_found():
    """A (0,0) grounding is the model's 'not found' answer: stop, don't click
    the corner (which would trip pyautogui's FAILSAFE)."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(point=(0.0, 0.0)), RecClicker()
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []  # never clicked the corner
    assert r["failed"]


def test_runner_deep_link_skips_clicking():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, dry_run=True,
                          use_deep_links=True, deep_link_settle_ms=0)
    eng, clk = StubEngine(), RecClicker()
    r, cb = _recorder()
    recipe = Recipe(name="d", utterances=("d",), steps=(Step("x"),),
                    deep_link="x-apple.systempreferences:com.apple.settings.Storage",
                    question="q?")
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert eng.ground_calls == []  # deep-link path: no per-step grounding
    assert r["answer"] == ["You have 120 GB available."]


def test_runner_expect_check_blocks_cascade_on_wrong_screen():
    """If the screen a step needs never appears, the runner stops safely
    instead of grounding (and clicking) on whatever is actually visible."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, answer_retry_ms=0)
    eng, clk = StubEngine(answer="no"), RecClicker()  # ask() always says not visible
    r, cb = _recorder()
    recipe = Recipe(name="v", utterances=("v",), steps=(
        Step("first", settle_ms=0),
        Step("second", settle_ms=0, expect="the Settings window"),
    ))
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert clk.calls == [(100.0, 100.0)]  # step 1 clicked; step 2 never did
    assert any("expectation not met" in f for f in r["failed"])
    assert len(eng.ask_calls) == 2  # checked, waited, rechecked once


def test_runner_expect_check_passes_when_screen_ready():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(answer="Yes."), RecClicker()
    r, cb = _recorder()
    recipe = Recipe(name="v", utterances=("v",), steps=(
        Step("first", settle_ms=0),
        Step("second", settle_ms=0, expect="the Settings window"),
    ))
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert clk.calls == [(100.0, 100.0), (100.0, 100.0)]  # both steps clicked
    assert not r["failed"]


def test_runner_expect_check_fails_open_on_unverifiable_reply():
    """A click-tuned model answering with a click tag cannot verify anything;
    the step must proceed (old behavior) rather than abort the task."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(answer="<click>10, 10</click>"), RecClicker()
    r, cb = _recorder()
    recipe = Recipe(name="v", utterances=("v",), steps=(
        Step("second", settle_ms=0, expect="the Settings window"),
    ))
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert clk.calls == [(100.0, 100.0)]
    assert not r["failed"]


def test_runner_answer_retries_when_pane_not_ready():
    """A 'the image does not show X' reply means the pane was still loading:
    the runner waits once and re-reads instead of failing the task."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, dry_run=True,
                          answer_retry_ms=0)
    eng = StubEngine(answer=[
        "The image does not show the macOS Storage settings screen.",
        "You have 120 GB available.",
    ])
    r, cb = _recorder()
    TaskRunner(cfg, eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert r["answer"] == ["You have 120 GB available."]
    assert len(eng.ask_calls) == 2


def test_runner_expect_check_blocks_on_sentence_negative():
    """Models rarely answer a literal 'no': 'The Settings window is not
    visible' must count as a negative too, not slip through as a yes."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, answer_retry_ms=0)
    eng = StubEngine(answer="The Settings window is not visible on this screen.")
    clk = RecClicker()
    r, cb = _recorder()
    recipe = Recipe(name="v", utterances=("v",), steps=(
        Step("first", settle_ms=0),
        Step("second", settle_ms=0, expect="the Settings window"),
    ))
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(recipe)
    assert clk.calls == [(100.0, 100.0)]  # step 2 never clicked
    assert any("expectation not met" in f for f in r["failed"])


def test_runner_off_screen_point_is_not_clicked():
    """A grounding answer outside the screen is a model failure, never a click."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0)
    eng, clk = StubEngine(point=(4000.0, 4000.0)), RecClicker()  # logical screen is 1000x1000
    r, cb = _recorder()
    TaskRunner(cfg, eng, clk, cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert clk.calls == []
    assert any("off-screen" in f for f in r["failed"])
    assert r["done"] == 1


def test_runner_abort_during_readback_stays_silent():
    """Cancelling while the model reads the final screen must not speak or show
    the answer afterwards - the cancelled result would be an unrequested action."""
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, dry_run=True,
                          answer_retry_ms=0)
    eng = StubEngine(answer="You have 120 GB available.")
    r, cb = _recorder()
    runner = TaskRunner(cfg, eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None)
    real_ask = eng.ask

    def ask_then_abort(image, prompt, *, max_tokens=128):
        out = real_ask(image, prompt, max_tokens=max_tokens)
        runner.abort()  # the hotkey lands while the model is reading
        return out

    eng.ask = ask_then_abort
    runner.run(RECIPE)
    assert r["answer"] == []
    assert "Cancelled" in r["status"] and "Done" not in r["status"]
    assert r["done"] == 1


def test_runner_answer_gives_honest_negative_after_retry():
    cfg = PointCastConfig(task_preview_ms=0, task_capture_hide_ms=0, dry_run=True,
                          answer_retry_ms=0)
    eng = StubEngine(answer=["I cannot see the storage information here."])
    r, cb = _recorder()
    TaskRunner(cfg, eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None).run(RECIPE)
    assert r["answer"] == ["I cannot see the storage information here."]
    assert len(eng.ask_calls) == 2  # tried twice, then reported honestly


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} task tests passed")
