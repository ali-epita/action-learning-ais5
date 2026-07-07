"""Headless tests for agent mode: reply parsing + the planned-click loop.

No Qt, no model: the runner is driven with a scripted engine (its ask() returns
predetermined agent replies), a recording clicker, and a callback recorder.
"""

from __future__ import annotations

import types

from ais5.pointcast.config import PointCastConfig
from ais5.pointcast.task.agent import AgentRunner, parse_agent_reply
from ais5.pointcast.task.runner import TaskCallbacks


# ── reply parsing ────────────────────────────────────────────────────────────
def test_parse_click_reply():
    kind, payload = parse_agent_reply("CLICK <click>500, 250</click> | the Wi-Fi icon")
    assert kind == "click"
    (pt, desc) = payload
    assert pt == (500.0, 250.0) and desc == "the Wi-Fi icon"


def test_parse_click_reply_is_lenient_about_tag_debris():
    # Observed live: the model dropped the opening tag.
    kind, ((x, y), desc) = parse_agent_reply("CLICK 644, 13</click> | Click Apple menu")
    assert kind == "click" and (x, y) == (644.0, 13.0) and desc == "Click Apple menu"
    assert parse_agent_reply("CLICK 100,200 | b")[0] == "click"
    assert parse_agent_reply("click(300, 400) | c")[0] == "click"


def test_parse_click_description_without_pipe():
    # Observed live: a dash instead of the pipe.
    kind, ((x, y), desc) = parse_agent_reply("CLICK 393, 390 — the Bluetooth option in the sidebar")
    assert kind == "click" and (x, y) == (393.0, 390.0)
    assert desc == "the Bluetooth option in the sidebar"


def test_parse_done_reply():
    assert parse_agent_reply("DONE | Dark mode is now on.") == ("done", "Dark mode is now on.")
    assert parse_agent_reply("done — opened the settings") == ("done", "opened the settings")


def test_parse_garbage_fails_safely():
    kind, _ = parse_agent_reply("I think you should maybe try the menu?")
    assert kind == "fail"
    assert parse_agent_reply("")[0] == "fail"


def test_click_takes_precedence_when_both_present():
    kind, _ = parse_agent_reply("CLICK <click>10, 10</click> | then we are DONE")
    assert kind == "click"


# ── runner loop ──────────────────────────────────────────────────────────────
class ScriptedAgentEngine:
    """ask() returns scripted replies in order; backend advertises a profile."""

    def __init__(self, replies, profile="norm1000"):
        self.replies = list(replies)
        self.calls = 0
        self.backend = types.SimpleNamespace(coord_profile=profile)

    def ask(self, image, prompt, *, max_tokens=128):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return r


class RecClicker:
    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append((round(x), round(y)))


class _IdMapper:
    ground_scale = 1.0
    logical_size = (1000, 500)

    def ground_to_logical(self, x, y):
        return (x, y)


def _capture():
    img = types.SimpleNamespace(size=(1000, 500))
    return types.SimpleNamespace(image=img, full_image=img, mapper=_IdMapper())


def _recorder():
    r = {"status": [], "say": [], "answer": [], "failed": [], "done": 0}
    cb = TaskCallbacks(
        status=r["status"].append,
        say=r["say"].append,
        answer=r["answer"].append,
        failed=r["failed"].append,
        done=lambda: r.__setitem__("done", r["done"] + 1),
    )
    return r, cb


def _cfg(**kw):
    defaults = {"task_preview_ms": 0, "task_capture_hide_ms": 0, "agent_settle_ms": 0}
    defaults.update(kw)
    return PointCastConfig(**defaults)


def test_agent_clicks_then_done_with_denormalized_coords():
    eng = ScriptedAgentEngine([
        "CLICK <click>500, 500</click> | the System Settings icon",
        "DONE | Dark mode is now enabled.",
    ])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("turn on dark mode")
    # norm1000 on a 1000x500 frame: (500, 500) -> (500, 250) pixels
    assert clk.calls == [(500, 250)]
    assert r["answer"] == ["Dark mode is now enabled."]
    assert r["done"] == 1 and not r["failed"]


def test_agent_absolute_profile_passthrough():
    eng = ScriptedAgentEngine(
        ["CLICK <click>320, 200</click> | a button", "DONE | done"], profile="absolute"
    )
    clk = RecClicker()
    _r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert clk.calls == [(320, 200)]


def test_agent_stops_on_unparseable_reply_without_clicking():
    eng = ScriptedAgentEngine(["hmm, not sure what to do here"])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert clk.calls == []
    assert r["failed"] and r["done"] == 1


def test_agent_step_cap():
    eng = ScriptedAgentEngine([
        "CLICK <click>100, 100</click> | a",
        "CLICK <click>200, 100</click> | b",
        "CLICK <click>300, 100</click> | c",
        "CLICK <click>400, 100</click> | d",
        "CLICK <click>500, 100</click> | e",
        "CLICK <click>600, 100</click> | f",
        "CLICK <click>700, 100</click> | never reached",
    ])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(agent_max_steps=6), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert len(clk.calls) == 6
    assert any("step cap" in f for f in r["failed"])


def test_agent_no_progress_guard():
    eng = ScriptedAgentEngine([
        "CLICK <click>500, 500</click> | the button",
        "CLICK <click>500, 500</click> | the button again",
    ])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert len(clk.calls) == 1  # second identical click is refused
    assert any("loop detected" in f for f in r["failed"])


def test_agent_catches_oscillating_loop():
    """A-B-A-B thrashing must be caught (the consecutive-only guard missed it)."""
    eng = ScriptedAgentEngine([
        "CLICK <click>100, 100</click> | A",
        "CLICK <click>800, 400</click> | B",
        "CLICK <click>100, 100</click> | A again",  # revisits A -> stop
        "CLICK <click>800, 400</click> | never reached",
    ])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert len(clk.calls) == 2  # A, B, then the third (A again) is refused
    assert any("loop detected" in f for f in r["failed"])


def test_agent_recovers_missing_description():
    """When a click reply omits its label, the runner asks for one so the run
    stays save-able as a recipe."""
    eng = ScriptedAgentEngine([
        "CLICK 500, 500",  # no "| description"
        "DONE | done",
    ])
    # the first ask is the plan; the second is the describe-target recovery
    eng.replies = ["CLICK 500, 500", "the Wi-Fi toggle", "DONE | done"]
    clk = RecClicker()
    _r, cb = _recorder()
    runner = AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None)
    runner.run("g")
    assert runner.clicked_targets == ["the Wi-Fi toggle"]


def test_agent_abort_during_preview_does_not_click():
    eng = ScriptedAgentEngine(["CLICK <click>500, 500</click> | x", "DONE | y"])
    clk = RecClicker()
    r, cb = _recorder()
    runner = AgentRunner(_cfg(task_preview_ms=50), eng, clk, cb, capture_fn=_capture, focus_fn=None)
    cb.point = lambda x, y: runner.abort()  # user hits the hotkey at the preview
    runner.run("g")
    assert clk.calls == []
    assert "Cancelled" in r["status"]


def test_agent_records_run_for_recipe_distillation():
    """A successful run keeps goal + the model's step descriptions, so the
    controller can save it as a replayable recipe."""
    eng = ScriptedAgentEngine([
        "CLICK <click>200, 200</click> | the gear icon in the Dock",
        "CLICK <click>500, 300</click> | Bluetooth in the sidebar",
        "DONE | Bluetooth settings are open.",
    ])
    _r, cb = _recorder()
    runner = AgentRunner(_cfg(), eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None)
    runner.run("open the bluetooth settings")
    assert runner.succeeded
    assert runner.goal == "open the bluetooth settings"
    assert runner.clicked_targets == ["the gear icon in the Dock", "Bluetooth in the sidebar"]


def test_agent_failed_run_is_not_marked_succeeded():
    eng = ScriptedAgentEngine(["gibberish with no click"])
    _r, cb = _recorder()
    runner = AgentRunner(_cfg(), eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None)
    runner.run("g")
    assert not runner.succeeded
    assert runner.clicked_targets == []


def test_agent_dry_run_never_clicks():
    eng = ScriptedAgentEngine(["CLICK <click>500, 500</click> | x", "DONE | y"])
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(dry_run=True), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert clk.calls == []
    assert r["answer"] == ["y"]


def test_agent_prompt_coordinate_note_matches_profile():
    """The prompt's coordinate instruction must match how the reply is decoded:
    prompting an absolute-profile model for 0-1000 coordinates would make a
    compliant answer land on the wrong pixel."""
    class PromptRecorder(ScriptedAgentEngine):
        def __init__(self, replies, profile):
            super().__init__(replies, profile)
            self.prompts = []

        def ask(self, image, prompt, *, max_tokens=128):
            self.prompts.append(prompt)
            return super().ask(image, prompt, max_tokens=max_tokens)

    eng = PromptRecorder(["DONE | ok"], profile="absolute")
    _r, cb = _recorder()
    AgentRunner(_cfg(), eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None).run("g")
    assert "(1000, 500)" in eng.prompts[0]  # the frame's pixel bottom-right
    assert "0 to 1000" not in eng.prompts[0]

    eng = PromptRecorder(["DONE | ok"], profile="norm1000")
    _r, cb = _recorder()
    AgentRunner(_cfg(), eng, RecClicker(), cb, capture_fn=_capture, focus_fn=None).run("g")
    assert "0 to 1000" in eng.prompts[0]


def test_agent_off_screen_plan_is_not_clicked():
    eng = ScriptedAgentEngine(["CLICK 4000, 900 | way out there"], profile="absolute")
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert clk.calls == []
    assert any("off-screen" in f for f in r["failed"])


def test_agent_degenerate_corner_plan_is_not_clicked():
    eng = ScriptedAgentEngine(["CLICK 0, 0 | mystery corner"], profile="absolute")
    clk = RecClicker()
    r, cb = _recorder()
    AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None).run("g")
    assert clk.calls == []
    assert any("degenerate" in f for f in r["failed"])


def test_agent_abort_during_thinking_discards_the_plan():
    """Cancelling while the model plans must swallow whatever it decided -
    even a spoken DONE must not surface after the user said stop."""
    eng = ScriptedAgentEngine(["DONE | Task finished."])
    clk = RecClicker()
    r, cb = _recorder()
    runner = AgentRunner(_cfg(), eng, clk, cb, capture_fn=_capture, focus_fn=None)
    real_ask = eng.ask

    def ask_then_abort(image, prompt, *, max_tokens=128):
        out = real_ask(image, prompt, max_tokens=max_tokens)
        runner.abort()
        return out

    eng.ask = ask_then_abort
    runner.run("g")
    assert r["answer"] == [] and clk.calls == []
    assert "Cancelled" in r["status"]
    assert runner.succeeded is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} agent tests passed")
