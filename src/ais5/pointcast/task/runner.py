"""Execute a recipe: capture -> ground -> click -> settle, step by step, then
read the answer off the final screen and speak it.

Runs entirely on the controller's single inference thread (all MLX work is
thread-bound), and reports progress through plain callbacks so the controller
can marshal them to the Qt UI thread and so the loop is headless-testable. No
planner model is involved: the recipe is the plan.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

from ...utils.logging import get_logger
from ..capture import capture_screen
from ..geometry import is_degenerate_point as _is_degenerate_point
from ..geometry import validate_logical_point
from .recipes import Recipe

_log = get_logger("pointcast.task")


def _noop(*_a, **_k) -> None:
    return None


_UNREADABLE_MARKERS = (
    "does not show", "doesn't show", "not show", "cannot see", "can't see",
    "not visible", "unable to", "no information", "still loading",
)


def _looks_unreadable(text: str) -> bool:
    """True when the model is telling us the screen was not what we expected
    (typically a pane that has not finished loading)."""
    low = text.lower()
    return any(m in low for m in _UNREADABLE_MARKERS)


_NEGATIVE_WORDS = {"no", "not", "cannot", "isn't", "can't", "nothing"}


def _is_negative_answer(reply: str) -> bool:
    """True when a yes/no verification reply is an explicit negative. This must
    catch more than a literal leading "no": models often answer in a sentence
    ("The Settings window is not visible"), and treating that as a yes would
    cascade clicks onto the wrong screen. Anything neither clearly yes nor
    clearly no stays non-negative (the caller deliberately fails open)."""
    words = re.findall(r"[a-z']+", reply.lower())
    if not words or words[0] == "yes":
        return False
    return words[0] == "no" or any(w in _NEGATIVE_WORDS for w in words)


@dataclass
class TaskCallbacks:
    status: Callable[[str], None] = _noop  # progress line ("Step 2/3: ...")
    say: Callable[[str], None] = _noop  # spoken narration
    point: Callable[[float, float], None] = _noop  # show the step crosshair
    clear: Callable[[], None] = _noop  # clear the crosshair
    # hide ALL of PointCast's overlays before a screenshot. Receives an optional
    # threading.Event the UI thread sets once the overlays are actually hidden,
    # so the runner waits for an acknowledgment instead of a blind sleep.
    hide: Callable[..., None] = _noop
    answer: Callable[[str], None] = _noop  # final answer read off the screen
    failed: Callable[[str], None] = _noop  # task aborted / could not continue
    done: Callable[[], None] = _noop  # always fires last (reset busy state)


class TaskRunner:
    def __init__(self, cfg, engine, clicker, cb: TaskCallbacks, *, capture_fn=None, focus_fn=None):
        self.cfg = cfg
        self.engine = engine
        self.clicker = clicker
        self.cb = cb
        self._capture = capture_fn or (lambda: capture_screen(cfg.ground_max_side_or_none, cfg.monitor_index))
        self._focus = focus_fn  # window_focus.focus_window_at, or None to skip
        self._aborted = False

    def abort(self) -> None:
        self._aborted = True

    def run(self, recipe: Recipe) -> None:
        try:
            self._run(recipe)
        except Exception as e:  # noqa: BLE001
            _log.exception("task crashed")
            self.cb.failed(repr(e))
        finally:
            self.cb.done()

    # ── the loop ──────────────────────────────────────────────────────────────
    def _run(self, recipe: Recipe) -> None:
        _log.info("task start: %s", recipe.name)
        self.cb.say(f"Okay. {recipe.name}.")

        if recipe.deep_link and self.cfg.use_deep_links:
            self._open_deep_link(recipe.deep_link)
        elif not self._navigate(recipe):
            return  # a step failed; _navigate already reported it

        if self._aborted:
            self.cb.status("Cancelled")
            return

        if recipe.question:
            self._answer(recipe.question)
        self.cb.status("Cancelled" if self._aborted else "Done")

    def _grab(self):
        """Hide PointCast's own overlays, wait for the UI thread to confirm they
        are gone (with a timeout fallback), then screenshot, so the model grounds
        your screen and never our status banner or crosshair."""
        hidden = threading.Event()
        try:
            self.cb.hide(hidden)
            hidden.wait(timeout=1.0)
        except TypeError:  # older callback without the ack parameter: no handshake
            self.cb.hide()
        self._sleep_ms(self.cfg.task_capture_hide_ms)  # window-server settle
        return self._capture()

    def _verify_expectation(self, expect: str) -> bool:
        """Ask the model whether ``expect`` is visible; on an explicit 'no',
        wait and recheck once (apps launch slowly). Fails OPEN: a reply that is
        neither yes nor no (e.g. a click-tuned model emitting a click tag)
        cannot verify anything, so the step proceeds as it always did."""
        for attempt in (1, 2):
            frame = self._grab()
            self.cb.status(f"Checking for {expect}…")
            reply = (self.engine.ask(
                frame.image,
                f"Look at this screenshot. Is {expect} visible? Answer with only yes or no.",
                max_tokens=6,
            ) or "").strip().lower()
            _log.info("    expect %r -> %r", expect, reply)
            if not _is_negative_answer(reply):
                return True  # yes, or unverifiable -> proceed
            if self._aborted or attempt == 2:
                return False
            self._sleep_ms(getattr(self.cfg, "answer_retry_ms", 2500))
        return False

    def _navigate(self, recipe: Recipe) -> bool:
        n = len(recipe.steps)
        for i, step in enumerate(recipe.steps, start=1):
            if self._aborted:
                self.cb.status("Cancelled")
                return False
            expect = getattr(step, "expect", "")
            if expect and not self._verify_expectation(expect):
                # The screen this step needs never appeared (previous click
                # missed, or the app did not open): stop instead of cascading
                # clicks onto the wrong window.
                self.cb.say("The screen I expected did not appear, so I stopped.")
                self.cb.failed(f"expectation not met before step {i}: {expect!r}")
                return False
            frame = self._grab()  # clean screenshot (our overlays hidden)
            self.cb.status(f"Step {i} of {n}: {step.target}")
            # No disambiguation UI in task mode: skip the stochastic candidate
            # sampling an uncertain step would otherwise pay for and discard.
            res = self.engine.ground(
                frame.image, step.target, want_candidates=False,
                full_image=getattr(frame, "full_image", None),
                full_scale=getattr(frame.mapper, "ground_scale", 1.0),
            )
            if res.point is not None and _is_degenerate_point(res.point):
                # A click at (0,0)/the extreme corner is the model's "I don't
                # know" answer, not a real target. Clicking it trips pyautogui's
                # corner FAILSAFE and crashes the task; treat it as not-found.
                _log.info("    step %d degenerate point %s for %r -> not found", i, res.point, step.target)
                res.point = None
            if res.point is None or not res.accepted:
                _log.info("    step %d uncertain for %r (accepted=%s)", i, step.target, res.accepted)
                self.cb.say("I am not sure where to click, so I stopped.")
                self.cb.failed(f"uncertain at step {i}: {step.target!r}")
                return False
            lx, ly = frame.mapper.ground_to_logical(*res.point)
            checked = validate_logical_point(lx, ly, frame.mapper.logical_size)
            if checked is None:
                _log.info("    step %d point (%.0f, %.0f) is off-screen for %r", i, lx, ly, step.target)
                self.cb.say("I am not sure where to click, so I stopped.")
                self.cb.failed(f"off-screen point at step {i}: {step.target!r}")
                return False
            lx, ly = checked
            _log.info("    step %d -> click (%.0f, %.0f)", i, lx, ly)
            self.cb.point(lx, ly)
            if step.say:
                self.cb.say(step.say)
            self._sleep_ms(self.cfg.task_preview_ms)  # let the user see/abort
            if self._aborted:  # the preview window is the user's veto: honor it
                self.cb.status("Cancelled")
                self.cb.clear()
                return False
            if not self.cfg.dry_run:
                if self._focus is not None:
                    activated = bool(self._focus(lx, ly, skip_pid=os.getpid()))
                    if activated:  # raised a background app: let it become key
                        self._sleep_ms(self.cfg.activate_settle_ms)
                if self._aborted:
                    self.cb.status("Cancelled")
                    self.cb.clear()
                    return False
                self.clicker.click(lx, ly)
            self._sleep_ms(step.settle_ms)
            self.cb.clear()
        return True

    def _open_deep_link(self, url: str) -> None:
        self.cb.status("Opening the settings pane")
        _log.info("    deep-link: %s", url)
        if not self.cfg.dry_run:
            subprocess.run(["open", url], check=False)
        # Settings panes (Storage especially) load and calculate for several
        # seconds; screenshotting too early reads a half-loaded pane.
        self._sleep_ms(getattr(self.cfg, "deep_link_settle_ms", 4000))

    def _answer(self, question: str) -> None:
        text = ""
        for attempt in (1, 2):
            frame = self._grab()  # clean screenshot for the read-back too
            self.cb.status("Reading the screen")
            # Short answer budget: the recipes ask for one sentence; a long
            # negative ramble costs 30s of dead air on stage.
            text = (self.engine.ask(frame.image, question, max_tokens=60) or "").strip()
            if self._aborted:
                return  # cancelled while the model was reading: stay silent
            if text and not _looks_unreadable(text):
                _log.info("    answer: %s", text)
                self.cb.answer(text)
                return
            if attempt == 1:  # the pane may still be loading/calculating
                _log.info("    screen not readable yet (%r); retrying once", text[:60])
                self.cb.status("Waiting for the screen to finish loading…")
                self._sleep_ms(getattr(self.cfg, "answer_retry_ms", 2500))
        if text:
            self.cb.answer(text)  # the honest negative beats silence
        else:
            self.cb.say("I could not read the result.")

    def _sleep_ms(self, ms: int) -> None:
        """Interruptible sleep so abort() takes effect promptly."""
        end = time.time() + max(0, ms) / 1000.0
        while time.time() < end:
            if self._aborted:
                return
            time.sleep(0.02)
