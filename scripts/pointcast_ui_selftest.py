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
    def __init__(self, result):
        self.result = result

    def warmup(self):
        pass

    def ground(self, image, instruction):
        return self.result


class RecClicker:
    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append((round(x, 1), round(y, 1)))

    def move(self, x, y):
        pass


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
    ctl.on_ground_done((GroundResult(point=None, badge="no match"), fake_mapper()))
    pump(app, 30)
    if not ctl.input_bar.isVisible():
        failures.append("no-match: did not reopen the input bar")
    ctl._abort()
    pump(app, 30)

    # ── scenario 3: uncertain -> disambiguate -> pick -> confirm -> click ──
    ctl._busy = True
    rec.calls.clear()
    result = GroundResult(point=(300, 200), accepted=False, badge="uncertain", candidates=[(100, 100), (300, 200)])
    ctl.on_ground_done((result, fake_mapper()))
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
    ctlr.on_target("the first button")  # demonstrate one step
    pump(app, 700)  # the step is also clicked (capture + ground + countdown)
    if ctlr._demo is None or ctlr._demo["steps"] != ["the first button"]:
        failures.append(f"record: step not captured ({ctlr._demo})")
    ctlr.on_target("save recipe")
    if ctlr._demo is not None:
        failures.append("record: demo not cleared after save")
    from ais5.pointcast.task import all_recipes, load_user_recipes, match_recipe

    saved = load_user_recipes(recpath)
    if not (len(saved) == 1 and saved[0].name == "open thing"
            and [s.target for s in saved[0].steps] == ["the first button"]):
        failures.append(f"record: saved recipe wrong ({saved})")
    if match_recipe("open thing", all_recipes(recpath)) is None:
        failures.append("record: saved recipe not matchable after save")
    ctlr.shutdown()

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("SELF-TEST PASSED  (no-match, accepted full path, and disambiguation pick all wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
