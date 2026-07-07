#!/usr/bin/env python3
"""Headless self-test for the PointCast UI + controller wiring.

Runs Qt under the 'offscreen' platform with a stub engine, a fake screen
capture, and a recording clicker, so the whole state machine is exercised with
no display, no macOS permissions, and no model:

  1. no-match  -> refuse-and-ask (reopens input bar)
  2. accepted  -> capture -> ground -> overlay -> HUD -> countdown -> click
  3. uncertain -> numbered candidates -> pick -> confirm -> click
  4. voice     -> gated on the bar; PTT/mic dictate into the field; Enter clicks

    .venv-demo/bin/python scripts/pointcast_ui_selftest.py
"""

from __future__ import annotations

import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AIS5_LOG_LEVEL", "WARNING")  # keep self-test output quiet

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ais5.pointcast import controller as controller_mod  # noqa: E402
from ais5.pointcast.config import PointCastConfig  # noqa: E402
from ais5.pointcast.engine import GroundResult  # noqa: E402
from ais5.pointcast.geometry import CoordinateMapper  # noqa: E402

SIZE = (1280, 800)


def fake_mapper():
    return CoordinateMapper(capture_size=SIZE, logical_size=SIZE, ground_scale=1.0)


def make_fake_capture():
    frame = types.SimpleNamespace(
        image=Image.new("RGB", SIZE, "white"),
        full_image=Image.new("RGB", SIZE, "white"),
        mapper=fake_mapper(),
    )
    return lambda *a, **k: frame


class StubEngine:
    def __init__(self, result, answer="Your total is 19.77 out of 20."):
        self.result = result
        self._answer = answer
        self.ask_prompts = []

    def warmup(self):
        return True, ""

    def ground(self, image, instruction, **kwargs):
        return self.result

    def ask(self, image, prompt, *, max_tokens=128):
        self.ask_prompts.append(prompt)
        return self._answer


class RecClicker:
    def __init__(self):
        self.calls = []
        self.typed = []
        self.enters = 0

    def click(self, x, y):
        self.calls.append((round(x, 1), round(y, 1)))

    def move(self, x, y):
        pass

    def type_text(self, text, **kw):
        self.typed.append(text)

    def press_enter(self):
        self.enters += 1


class StubRecorder:
    def __init__(self, samples):
        self.samples = samples

    def start(self):
        pass

    def stop(self):
        return self.samples


class StubSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, samples):
        return self.text


def pump(app, ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def main() -> int:
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    cfg = PointCastConfig(dry_run=True, tts_engine="off", countdown_seconds=0.3, speak_locator=False)
    failures: list[str] = []

    # ── scenario 1: accepted single point, full async path ──
    controller_mod.capture_screen = make_fake_capture()
    ctl = controller_mod.Controller(cfg, StubEngine(GroundResult(point=(200, 150), accepted=True, badge="locked")))
    rec = RecClicker()
    ctl.clicker = rec

    ctl.on_target("the search box")
    pump(app, 240)  # capture(160) + ground
    if not (ctl.overlay.isVisible() and ctl.hud.isVisible() and ctl._pending):
        failures.append("accepted: overlay/HUD/pending not set after grounding")
    pump(app, 700)  # countdown(300) + click(80)
    if rec.calls != [(200.0, 150.0)]:
        failures.append(f"accepted: expected click at (200,150), got {rec.calls}")
    if ctl._busy:
        failures.append("accepted: still busy after click")

    # ── scenario 2: no-match re-prompts ──
    ctl._busy = False
    ctl.on_ground_done((ctl._ground_gen, GroundResult(point=None, badge="no match"), fake_mapper()))
    pump(app, 30)
    if not ctl.input_bar.isVisible():
        failures.append("no-match: did not reopen the input bar")
    ctl._abort()
    pump(app, 30)

    # ── scenario 3: uncertain -> disambiguate -> pick -> confirm -> click ──
    ctl._busy = True
    rec.calls.clear()
    result = GroundResult(point=(300, 200), accepted=False, badge="uncertain", candidates=[(100, 100), (300, 200)])
    ctl.on_ground_done((ctl._ground_gen, result, fake_mapper()))
    pump(app, 40)
    if not (ctl.disambig.isVisible() and ctl.overlay.isVisible()):
        failures.append("disambig: markers/HUD not shown")
    ctl.disambig.picked.emit(1)  # pick candidate #2 -> (300,200)
    pump(app, 40)
    if ctl._pending != (300.0, 200.0):
        failures.append(f"disambig: pending after pick = {ctl._pending}, expected (300,200)")
    if not ctl.hud.isVisible():
        failures.append("disambig: confirm HUD not shown after pick")
    ctl.hud.confirmed.emit()  # simulate Enter
    pump(app, 200)
    if rec.calls != [(300.0, 200.0)]:
        failures.append(f"disambig: expected click at (300,200), got {rec.calls}")
    ctl.shutdown()

    # ── scenario 1b: READ verb -> speaks + shows result card ──
    ctlr = controller_mod.Controller(cfg, StubEngine(GroundResult(point=(10, 10)), answer="You scored 19.77."))
    ctlr.clicker = RecClicker()
    ctlr.on_target("read my final kaggle grade")
    pump(app, 260)  # capture delay + ask
    if not ctlr.result_card.isVisible():
        failures.append("read: result card not shown")
    if ctlr.result_card._answer.text() != "You scored 19.77.":
        failures.append(f"read: wrong answer shown ({ctlr.result_card._answer.text()!r})")
    if ctlr._busy:
        failures.append("read: still busy after answer")
    if not ctlr.engine.ask_prompts or "kaggle grade" not in ctlr.engine.ask_prompts[0]:
        failures.append("read: VQA prompt did not include the query")
    ctlr.shutdown()

    # ── scenario 1c: TYPE verb -> ground field, confirm, click + type (+enter) ──
    ctt = controller_mod.Controller(cfg, StubEngine(GroundResult(point=(120, 90), accepted=True, badge="locked")))
    rct = RecClicker()
    ctt.clicker = rct
    ctt.on_target("type ali.cherri into the username field and press enter")
    pump(app, 260)  # capture + ground -> confirm HUD
    if ctt._pending_action.get("kind") != "type":
        failures.append(f"type: pending action not type ({ctt._pending_action})")
    if not ctt.hud.isVisible():
        failures.append("type: confirm HUD not shown before typing")
    ctt.hud.confirmed.emit()
    pump(app, 400)  # click(80) + focus settle(180) + type
    if rct.calls != [(120.0, 90.0)]:
        failures.append(f"type: field not clicked, got {rct.calls}")
    if rct.typed != ["ali.cherri"]:
        failures.append(f"type: wrong text typed ({rct.typed})")
    if rct.enters != 1:
        failures.append(f"type: enter not pressed ({rct.enters})")
    ctt.shutdown()

    # ── scenario 1d: TYPE without target -> confirm then type into focused ──
    ctf = controller_mod.Controller(cfg, StubEngine(GroundResult(point=(0, 0))))
    rcf = RecClicker()
    ctf.clicker = rcf
    ctf.on_target("type hello world")
    pump(app, 60)
    if not ctf.hud.isVisible():
        failures.append("type-focused: confirm HUD not shown")
    ctf.hud.confirmed.emit()
    pump(app, 200)
    if rcf.calls:
        failures.append(f"type-focused: should not click a field, got {rcf.calls}")
    if rcf.typed != ["hello world"]:
        failures.append(f"type-focused: wrong text typed ({rcf.typed})")
    ctf.shutdown()

    # ── scenario 4: voice is gated on the search bar, dictates into the field ──
    cfg_v = PointCastConfig(
        dry_run=True, tts_engine="off", countdown_seconds=0.3, speak_locator=False, enable_voice=True
    )
    ctlv = controller_mod.Controller(cfg_v, StubEngine(GroundResult(point=(640, 400), accepted=True, badge="locked")))
    recv = RecClicker()
    ctlv.clicker = recv
    ctlv.recorder = StubRecorder(np.zeros(16000, dtype="float32"))
    ctlv.stt = StubSTT("the send button")

    # PTT does nothing until the user opens the bar with the hotkey
    ctlv.on_ptt_press()
    if ctlv._recording:
        failures.append("voice: PTT recorded while the search bar was closed (should be gated)")

    ctlv.start_interaction()  # hotkey -> search bar
    pump(app, 30)
    if not ctlv.input_bar.isVisible():
        failures.append("voice: search bar not shown on hotkey")

    # hold PTT -> listening animation on; release -> transcript fills the field (no click yet)
    ctlv.on_ptt_press()
    if not (ctlv._recording and ctlv.input_bar._mic is not None and ctlv.input_bar._mic._listening):
        failures.append("voice: PTT press did not start the listening animation")
    ctlv.on_ptt_release()
    pump(app, 600)
    if ctlv.input_bar.text() != "the send button":
        failures.append(f"voice: transcript not dictated into the field, got {ctlv.input_bar.text()!r}")
    if recv.calls:
        failures.append(f"voice: clicked before the user confirmed, got {recv.calls}")

    # the mic button toggles recording the same way (start, then stop)
    ctlv.on_mic_clicked()
    if not ctlv._recording:
        failures.append("voice: mic button did not start recording")
    ctlv.on_mic_clicked()
    pump(app, 600)

    # Enter on the field -> ground -> countdown -> click
    ctlv.input_bar._submit()
    pump(app, 1300)
    if recv.calls != [(640.0, 400.0)]:
        failures.append(f"voice: expected click at (640,400) after Enter, got {recv.calls}")
    ctlv.shutdown()

    # ── scenario 5: record-by-demonstration -> save -> matchable ──
    import os
    import tempfile

    recpath = os.path.join(tempfile.mkdtemp(), "recipes.json")
    cfg_r = PointCastConfig(
        dry_run=True, tts_engine="off", countdown_seconds=0.2, speak_locator=False,
        enable_tasks=True, recipes_path=recpath,
    )
    ctlr = controller_mod.Controller(cfg_r, StubEngine(GroundResult(point=(300, 300), accepted=True, badge="locked")))
    ctlr.clicker = RecClicker()
    ctlr.on_target("record open thing")
    if ctlr._demo is None or ctlr._demo["name"] != "open thing":
        failures.append("record: did not start a demo session")
    if not ctlr.record_hud.isVisible():
        failures.append("record: Save/Cancel HUD not shown when recording starts")
    ctlr.on_target("the first button")  # demonstrate one step
    pump(app, 60)  # the record HUD must be HIDDEN during the capture/ground
    if ctlr.record_hud.isVisible():
        failures.append("record: Save/Cancel HUD not hidden during the step screenshot")
    pump(app, 700)  # the step is also clicked (capture + ground + countdown)
    if ctlr._demo is None or ctlr._demo["steps"] != ["the first button"]:
        failures.append(f"record: step not captured ({ctlr._demo})")
    if not ctlr.record_hud.isVisible():
        failures.append("record: Save/Cancel HUD not restored after the step")
    # a step the user retries (R at the confirm HUD) must NOT stay recorded
    ctlr.on_target("the wrong button")
    pump(app, 240)  # capture + ground -> confirm HUD up, countdown running
    ctlr.on_retry()
    pump(app, 30)
    if ctlr._demo is None or ctlr._demo["steps"] != ["the first button"]:
        failures.append(f"record: retried step leaked into the recording ({ctlr._demo})")
    ctlr._abort()  # close the reopened input bar
    pump(app, 30)

    # a step the user cancels (Esc during the countdown) must not stay either
    ctlr._busy = False
    ctlr.on_target("another wrong one")
    pump(app, 240)
    ctlr._abort()
    pump(app, 30)
    if ctlr._demo is None or ctlr._demo["steps"] != ["the first button"]:
        failures.append(f"record: cancelled step leaked into the recording ({ctlr._demo})")

    # the Save button is equivalent to the "save recipe" command (no read-back)
    if not ctlr.record_hud.isVisible():
        failures.append("record: Save/Cancel HUD not visible before saving")
    ctlr.on_target("save recipe and read the free storage")
    if ctlr.record_hud.isVisible():
        failures.append("record: Save/Cancel HUD not hidden after save")
    if ctlr._demo is not None:
        failures.append("record: demo not cleared after save")
    from ais5.pointcast.task import all_recipes, load_user_recipes, match_recipe

    saved = load_user_recipes(recpath)
    if not (len(saved) == 1 and saved[0].name == "open thing"
            and [s.target for s in saved[0].steps] == ["the first button"]):
        failures.append(f"record: saved recipe wrong ({saved})")
    if not saved or "free storage" not in saved[0].question:
        failures.append(f"record: read-back question not saved ({saved[0].question if saved else None!r})")
    if match_recipe("open thing", all_recipes(recpath)) is None:
        failures.append("record: saved recipe not matchable after save")
    ctlr.shutdown()

    # ── scenario 5a2: the Save/Cancel BUTTONS drive the same save/cancel ──
    recpath2 = os.path.join(tempfile.mkdtemp(), "recipes.json")
    cfg_b = PointCastConfig(dry_run=True, tts_engine="off", countdown_seconds=0.2,
                            speak_locator=False, enable_tasks=True, recipes_path=recpath2)
    ctlb = controller_mod.Controller(cfg_b, StubEngine(GroundResult(point=(50, 50), accepted=True, badge="locked")))
    ctlb.clicker = RecClicker()
    # Cancel button aborts the recording
    ctlb.on_target("record thing one")
    ctlb.record_hud.cancel.emit()
    pump(app, 30)
    if ctlb._demo is not None or ctlb.record_hud.isVisible():
        failures.append("record-buttons: Cancel button did not stop the recording")
    # Save button saves the demonstrated steps
    ctlb.on_target("record thing two")
    ctlb.on_target("a button")
    pump(app, 500)
    ctlb.record_hud.save.emit()
    pump(app, 30)
    if ctlb._demo is not None:
        failures.append("record-buttons: Save button did not clear the demo")
    saved_b = load_user_recipes(recpath2)
    if not (len(saved_b) == 1 and saved_b[0].name == "thing two"):
        failures.append(f"record-buttons: Save button did not save the recipe ({saved_b})")
    ctlb.shutdown()

    # ── scenario 5b: agent run -> "save recipe as" -> replayable by name ──
    import types as _types

    class StubAgentEngine:
        backend = _types.SimpleNamespace(coord_profile="absolute")

        def __init__(self):
            self.replies = [
                "CLICK 200, 200 | the gear icon in the Dock",
                "DONE | Settings is open.",
            ]
            self.n = 0

        def warmup(self):
            return True, ""

        def ground(self, image, instruction, **kw):
            return GroundResult(point=(10, 10), accepted=True, badge="locked")

        def ask(self, image, prompt, *, max_tokens=128):
            r = self.replies[min(self.n, len(self.replies) - 1)]
            self.n += 1
            return r

    agent_recpath = os.path.join(tempfile.mkdtemp(), "recipes.json")
    cfg_a = PointCastConfig(
        dry_run=True, tts_engine="off", speak_locator=False,
        enable_agent=True, enable_tasks=True, recipes_path=agent_recpath,
        task_preview_ms=0, task_capture_hide_ms=0, agent_settle_ms=0,
    )
    ctla = controller_mod.Controller(cfg_a, StubAgentEngine())
    ctla.clicker = RecClicker()
    ctla.on_target("do open the settings")
    pump(app, 2500)  # agent run completes on the worker thread (real screen grabs)
    if ctla._last_agent_run is None:
        failures.append("agent-save: successful run was not recorded")
    ctla.on_target("save recipe as quick settings")
    pump(app, 60)
    from ais5.pointcast.task import all_recipes as _all, match_recipe as _match

    saved_names = {r.name for r in _all(agent_recpath)}
    if "quick settings" not in saved_names:
        failures.append(f"agent-save: recipe not saved ({saved_names})")
    if _match("open the settings", _all(agent_recpath)) is None:
        failures.append("agent-save: original goal phrase does not match the saved recipe")
    ctla.shutdown()

    # ── scenario 5c: settings panel toggles apply live and persist ──
    setpath = os.path.join(tempfile.mkdtemp(), "settings.json")
    cfg_s2 = PointCastConfig(dry_run=True, tts_engine="off", speak_locator=False,
                             settings_path=setpath)
    ctls2 = controller_mod.Controller(cfg_s2, StubEngine(GroundResult(point=(5, 5))))
    ctls2.clicker = RecClicker()
    ctls2.on_setting_changed("use_deep_links", True)
    ctls2.on_setting_changed("enable_voice", True)
    pump(app, 30)
    if not (cfg_s2.use_deep_links and cfg_s2.enable_voice):
        failures.append("settings: toggles did not apply to the config")
    if ctls2.recorder is None or ctls2.stt is None:
        failures.append("settings: enabling voice did not create recorder/STT")
    if not ctls2.input_bar._mic.isVisibleTo(ctls2.input_bar):
        failures.append("settings: mic button not shown after enabling voice")
    from ais5.pointcast.settings import load_settings as _load_set

    persisted = _load_set(setpath)
    if not (persisted.get("use_deep_links") and persisted.get("enable_voice")):
        failures.append(f"settings: not persisted ({persisted})")
    ctls2.on_setting_changed("tts_engine", "off")
    if ctls2.speaker.engine != "off":
        failures.append("settings: tts engine change not applied to the speaker")
    ctls2.shutdown()

    # ── scenario 6: cancelled dictation must NOT resurrect as an action ──
    cfg_s = PointCastConfig(
        dry_run=True, tts_engine="off", countdown_seconds=0.2, speak_locator=False, enable_voice=True
    )
    ctls = controller_mod.Controller(cfg_s, StubEngine(GroundResult(point=(50, 50), accepted=True, badge="locked")))
    recs = RecClicker()
    ctls.clicker = recs
    ctls.recorder = StubRecorder(np.zeros(16000, dtype="float32"))

    class SlowSTT:
        def transcribe(self, samples):
            time.sleep(0.25)  # transcription still in flight when the user cancels
            return "check my storage"

    ctls.stt = SlowSTT()
    ctls.start_interaction()
    pump(app, 30)
    ctls.on_mic_clicked()  # start
    ctls.on_mic_clicked()  # stop -> SlowSTT running on the inference thread
    ctls._abort()  # user hits Esc while transcribing
    pump(app, 600)  # transcription completes after the cancel
    if ctls._busy or recs.calls or ctls.hud.isVisible() or ctls.overlay.isVisible():
        failures.append("stale dictation: cancelled speech still triggered an action")
    ctls.shutdown()

    # ── scenario 7: live voice loop — speak, act, listen again, end on "thanks" ──
    class StubLiveListener:
        def __init__(self, **kw):
            self.on_utterance = kw.get("on_utterance")
            self.starts = 0
            self._run = False

        def start(self):
            self.starts += 1
            self._run = True

        def stop(self):
            self._run = False

        @property
        def running(self):
            return self._run

    class SeqSTT:
        def __init__(self, texts):
            self.texts = list(texts)
            self.i = 0

        def load(self):
            return self

        def transcribe(self, samples):
            t = self.texts[min(self.i, len(self.texts) - 1)]
            self.i += 1
            return t

    real_listener = controller_mod.LiveListener
    controller_mod.LiveListener = StubLiveListener
    try:
        cfg_l = PointCastConfig(dry_run=True, tts_engine="off", countdown_seconds=0.2,
                                speak_locator=False, enable_voice=True, enable_live_voice=True)
        ctll = controller_mod.Controller(cfg_l, StubEngine(GroundResult(point=(80, 80), accepted=True, badge="locked")))
        ctll.clicker = RecClicker()
        ctll.recorder = StubRecorder(np.zeros(16000, dtype="float32"))
        ctll.stt = SeqSTT(["the send button", "thank you"])
        # hotkey opens the ORIGINAL command bar (not live)
        ctll.start_interaction()
        pump(app, 40)
        if not ctll.input_bar.isVisible() or ctll._live_active:
            failures.append("live: hotkey should open the command bar, not the live session")
        # live starts explicitly via the ◉ button
        ctll.input_bar.live_clicked.emit()
        pump(app, 400)  # the mic opens after a TTS settle
        if not ctll._live_active:
            failures.append("live: ◉ button did not start the session")
        if not ctll.live_hud.isVisible():
            failures.append("live: LiveHUD not shown on session start")
        if ctll.input_bar.isVisible():
            failures.append("live: command bar still shown after entering live")
        lis = ctll._live_listener
        if lis is None or lis.starts != 1:
            failures.append(f"live: listener not started ({getattr(lis, 'starts', None)})")
        # first spoken command -> command runs -> mic reopens after the click
        ctll.sig.live_utterance.emit(np.zeros(16000, dtype="float32"))
        pump(app, 900)  # transcribe + capture + ground + countdown + dry click
        if ctll.clicker.calls != [(80.0, 80.0)]:
            failures.append(f"live: spoken command did not click ({ctll.clicker.calls})")
        if "the send button" not in ctll.live_hud._history.text():
            failures.append("live: transcript turn not shown in the LiveHUD history")
        if lis.starts < 2:
            failures.append(f"live: mic did not reopen after the action (starts={lis.starts})")
        if not ctll.live_hud.isVisible():
            failures.append("live: LiveHUD not restored after the action")
        # "thank you" ends the session
        ctll.sig.live_utterance.emit(np.zeros(16000, dtype="float32"))
        pump(app, 400)
        if ctll._live_active:
            failures.append("live: 'thank you' did not end the session")
        if ctll.live_hud.isVisible():
            failures.append("live: LiveHUD still visible after session end")
        ctll.shutdown()
    finally:
        controller_mod.LiveListener = real_listener

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("SELF-TEST PASSED  (no-match, accepted full path, and disambiguation pick all wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
