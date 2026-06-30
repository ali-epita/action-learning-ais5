"""Execute a recipe: capture -> ground -> click -> settle, step by step, then
read the answer off the final screen and speak it.

Runs entirely on the controller's single inference thread (all MLX work is
thread-bound), and reports progress through plain callbacks so the controller
can marshal them to the Qt UI thread and so the loop is headless-testable. No
planner model is involved: the recipe is the plan.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from ...utils.logging import get_logger
from ..capture import capture_screen
from .recipes import Recipe

_log = get_logger("pointcast.task")


def _noop(*_a, **_k) -> None:
    return None


@dataclass
class TaskCallbacks:
    status: Callable[[str], None] = _noop  # progress line ("Step 2/3: ...")
    say: Callable[[str], None] = _noop  # spoken narration
    point: Callable[[float, float], None] = _noop  # show the step crosshair
    clear: Callable[[], None] = _noop  # clear the crosshair
    hide: Callable[[], None] = _noop  # hide ALL of PointCast's overlays before a screenshot
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
        self.cb.status("Done")

    def _grab(self):
        """Hide PointCast's own overlays, let them disappear, then screenshot, so
        the model grounds your screen and never our status banner or crosshair."""
        self.cb.hide()
        self._sleep_ms(self.cfg.task_capture_hide_ms)
        return self._capture()

    def _navigate(self, recipe: Recipe) -> bool:
        n = len(recipe.steps)
        for i, step in enumerate(recipe.steps, start=1):
            if self._aborted:
                self.cb.status("Cancelled")
                return False
            frame = self._grab()  # clean screenshot (our overlays hidden)
            self.cb.status(f"Step {i} of {n}: {step.target}")
            res = self.engine.ground(frame.image, step.target)
            if res.point is None or not res.accepted:
                _log.info("    step %d uncertain for %r (accepted=%s)", i, step.target, res.accepted)
                self.cb.say("I am not sure where to click, so I stopped.")
                self.cb.failed(f"uncertain at step {i}: {step.target!r}")
                return False
            lx, ly = frame.mapper.ground_to_logical(*res.point)
            _log.info("    step %d -> click (%.0f, %.0f)", i, lx, ly)
            self.cb.point(lx, ly)
            if step.say:
                self.cb.say(step.say)
            self._sleep_ms(self.cfg.task_preview_ms)  # let the user see/abort
            if not self.cfg.dry_run:
                if self._focus is not None:
                    self._focus(lx, ly, skip_pid=os.getpid())
                self.clicker.click(lx, ly)
            self._sleep_ms(step.settle_ms)
            self.cb.clear()
        return True

    def _open_deep_link(self, url: str) -> None:
        self.cb.status("Opening the settings pane")
        _log.info("    deep-link: %s", url)
        if not self.cfg.dry_run:
            subprocess.run(["open", url], check=False)
        self._sleep_ms(1200)

    def _answer(self, question: str) -> None:
        frame = self._grab()  # clean screenshot for the read-back too
        self.cb.status("Reading the screen")
        text = (self.engine.ask(frame.image, question) or "").strip()
        if text:
            _log.info("    answer: %s", text)
            self.cb.answer(text)
        else:
            self.cb.say("I could not read the result.")

    def _sleep_ms(self, ms: int) -> None:
        """Interruptible sleep so abort() takes effect promptly."""
        end = time.time() + max(0, ms) / 1000.0
        while time.time() < end:
            if self._aborted:
                return
            time.sleep(0.02)
