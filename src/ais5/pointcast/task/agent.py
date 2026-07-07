"""Free-form agent mode: the model plans the next click toward a spoken goal.

Unlike recipes (curated, fixed sequences), the agent decides each click itself
from the current screenshot: grab -> "what is the single next click toward the
goal?" -> preview -> click -> settle -> repeat. It inherits the task runner's
entire safety envelope: every step shows the crosshair preview with the abort
window before dispatch, the hotkey aborts between and during steps, the loop is
hard-capped at ``agent_max_steps``, and an unparseable or non-progressing reply
stops the task instead of guessing.

Runs on the controller's single inference thread and reports through the same
``TaskCallbacks``, so the controller wiring is identical to recipes.
"""

from __future__ import annotations

import os
import re
from math import hypot

from ...prompt.action import parse_click
from ...utils.logging import get_logger
from ..geometry import is_degenerate_point, validate_logical_point
from .runner import TaskCallbacks, TaskRunner

_log = get_logger("pointcast.agent")

AGENT_PROMPT = (
    "You are operating a macOS computer for the user. The image is the CURRENT "
    "state of the screen.\n"
    "Goal: {goal}\n"
    "Steps already taken:\n{history}\n"
    "Decide the SINGLE next mouse click that makes progress toward the goal.\n"
    "Reply with EXACTLY ONE line, either a click or done, like these examples:\n"
    "CLICK 512, 84 | the OK button in the dialog\n"
    "DONE | The task is finished.\n"
    "{coord_note}\n"
    "If the goal is already achieved, or no click can achieve it, reply DONE."
)

# The coordinate instruction MUST match the backend's coordinate profile: the
# reply is converted by _to_pixels according to that profile, so prompting an
# absolute-profile model for 0-1000 coordinates (or vice versa) would make a
# fully compliant answer land on the wrong pixel.
_COORD_NOTE_NORM1000 = (
    "Coordinates are integers from 0 to 1000, relative to the image: (0, 0) is "
    "the top-left corner and (1000, 1000) is the bottom-right corner."
)
_COORD_NOTE_ABSOLUTE = (
    "Coordinates are pixel positions on the image: (0, 0) is the top-left "
    "corner and ({w}, {h}) is the bottom-right corner."
)


def _coord_note(profile: str, size: tuple[int, int]) -> str:
    if profile == "norm1000":
        return _COORD_NOTE_NORM1000
    return _COORD_NOTE_ABSOLUTE.format(w=size[0], h=size[1])

_DONE_RE = re.compile(r"^\s*DONE\b\s*[|:—-]?\s*(.*)", re.IGNORECASE | re.MULTILINE)
# Lenient: models drift on the exact syntax ("CLICK 644, 13</click>", "CLICK
# <click>644,13", "click(644, 13)") — accept CLICK followed by two numbers with
# any tag debris in between.
_CLICK_RE = re.compile(
    r"\bCLICK\b[^0-9(-]*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)


def parse_agent_reply(text: str) -> tuple[str, object]:
    """-> ("done", message) | ("click", ((x, y), description)) | ("fail", reason).

    The click point is in the MODEL'S native coordinate space; the runner
    converts via the backend's coordinate profile.
    """
    text = (text or "").strip()
    if not text:
        return ("fail", "empty reply")
    if m := _CLICK_RE.search(text):
        # Description: whatever follows the pipe, or failing that whatever
        # follows the coordinates (models drift between "|", "—", ":").
        desc = text.split("|", 1)[1] if "|" in text else text[m.end():]
        desc = re.sub(r"^(?:\s+|</click>|[|:—–-]+)+", "", desc).strip()[:80]  # noqa: RUF001 — en dash is a real model output
        return ("click", ((float(m.group(1)), float(m.group(2))), desc))
    desc = text.split("|", 1)[1].strip()[:80] if "|" in text else ""
    m = _DONE_RE.search(text)
    if m and "<click>" not in text.lower():
        return ("done", m.group(1).strip())
    parsed = parse_click(text)
    if parsed.point is not None:
        return ("click", (parsed.point, desc))
    return ("fail", f"unparseable reply: {text[:60]!r}")


def _to_pixels(pt: tuple[float, float], size: tuple[int, int], profile: str) -> tuple[float, float]:
    if profile == "norm1000":
        return (pt[0] / 1000.0 * size[0], pt[1] / 1000.0 * size[1])
    return pt


class AgentRunner(TaskRunner):
    """Model-planned multi-step clicking. Reuses TaskRunner's capture/hide/
    abort/preview machinery; only the step source differs (model vs recipe)."""

    def __init__(self, cfg, engine, clicker, cb: TaskCallbacks, *, capture_fn=None, focus_fn=None):
        super().__init__(cfg, engine, clicker, cb, capture_fn=capture_fn, focus_fn=focus_fn)
        self._history: list[str] = []
        # Run record, for distilling a successful run into a saved recipe:
        self.goal: str = ""
        self.clicked_targets: list[str] = []  # the model's own step descriptions
        self.succeeded: bool = False  # True only when the run reached DONE

    def run(self, goal: str) -> None:  # type: ignore[override]
        self.goal = goal
        try:
            self._run_agent(goal)
        except Exception as e:  # noqa: BLE001
            _log.exception("agent task crashed")
            self.cb.failed(repr(e))
        finally:
            self.cb.done()

    def _describe_target(self, image, px: float, py: float) -> str:
        """Recover a short re-groundable label for the element at (px, py) in
        grounding-image pixels, when the model's plan omitted one."""
        w, h = image.size
        nx, ny = round(px / w * 1000), round(py / h * 1000)
        try:
            reply = self.engine.ask(
                image,
                f"In this screenshot, name the single UI element at position "
                f"({nx}, {ny}) on a 0-1000 scale, in 3-8 words, as a phrase "
                f"someone could use to find it again. Reply with only the phrase.",
                max_tokens=24,
            )
        except Exception:  # noqa: BLE001
            return ""
        return (reply or "").strip().strip('"').split("\n")[0][:80]

    def _run_agent(self, goal: str) -> None:
        _log.info("agent start: %r", goal)
        self.cb.say(f"Okay. {goal}.")
        profile = getattr(getattr(self.engine, "backend", None), "coord_profile", "absolute")
        max_steps = getattr(self.cfg, "agent_max_steps", 10)
        recent_clicks: list[tuple[float, float]] = []  # loop detection over a window

        for i in range(1, max_steps + 1):
            if self._aborted:
                self.cb.status("Cancelled")
                return
            frame = self._grab()
            self.cb.status(f"Thinking (step {i} of at most {max_steps})…")
            history = "\n".join(f"{n}. {s}" for n, s in enumerate(self._history, 1)) or "(none yet)"
            prompt = AGENT_PROMPT.format(
                goal=goal, history=history,
                coord_note=_coord_note(profile, frame.image.size),
            )
            text = self.engine.ask(frame.image, prompt, max_tokens=64)
            _log.info("    step %d reply: %r", i, (text or "")[:80])
            if self._aborted:
                # Cancelled while the model was thinking: whatever it decided
                # (a click OR a spoken DONE) must not surface now.
                self.cb.status("Cancelled")
                return
            kind, payload = parse_agent_reply(text)

            if kind == "done":
                msg = str(payload) or "Done."
                self.succeeded = True
                self.cb.answer(msg)
                return
            if kind == "fail":
                _log.info("    stopping: %s", payload)
                self.cb.say("I am not sure how to continue, so I stopped.")
                self.cb.failed(f"agent stopped at step {i}: {payload}")
                return

            (raw_pt, desc) = payload  # type: ignore[misc]
            px, py = _to_pixels(raw_pt, frame.image.size, profile)
            if is_degenerate_point((px, py)):
                self.cb.say("I am not sure where to click, so I stopped.")
                self.cb.failed(f"degenerate point at step {i}: {raw_pt}")
                return
            lx, ly = frame.mapper.ground_to_logical(px, py)
            checked = validate_logical_point(lx, ly, frame.mapper.logical_size)
            if checked is None:
                self.cb.say("That click would land off the screen, so I stopped.")
                self.cb.failed(f"off-screen point at step {i}: {raw_pt}")
                return
            lx, ly = checked
            # Loop detection: stop if this click revisits ANY of the recent
            # clicks, not just the immediately previous one. A thrashing model
            # oscillates A-B-A-B (each step "changes" the screen, so a
            # consecutive-only check never fires) — a windowed check catches it.
            if any(hypot(lx - qx, ly - qy) < 24 for (qx, qy) in recent_clicks):
                self.cb.say("I seem to be going in circles, so I stopped.")
                self.cb.failed(f"loop detected at step {i} (revisited a recent click)")
                return
            if not desc:
                # The model omitted the description (small models often drop it).
                # A step with no re-groundable label cannot be saved into a
                # recipe later, so recover one with a cheap targeted ask.
                desc = self._describe_target(frame.image, px, py)
            step_label = desc or f"step {i}"
            _log.info("    step %d -> click (%.0f, %.0f)  %s", i, lx, ly, step_label)
            self.cb.status(f"Step {i}: {step_label}")
            self.cb.point(lx, ly)
            if desc:
                self.cb.say(desc)
            self._sleep_ms(self.cfg.task_preview_ms)  # the user's veto window
            if self._aborted:
                self.cb.status("Cancelled")
                self.cb.clear()
                return
            if not self.cfg.dry_run:
                if self._focus is not None and bool(self._focus(lx, ly, skip_pid=os.getpid())):
                    self._sleep_ms(self.cfg.activate_settle_ms)
                if self._aborted:
                    self.cb.status("Cancelled")
                    self.cb.clear()
                    return
                self.clicker.click(lx, ly)
            recent_clicks.append((lx, ly))
            recent_clicks[:] = recent_clicks[-4:]  # window of the last 4 clicks
            self._history.append(step_label)
            self.clicked_targets.append(desc)  # "" when the model gave no description
            self._sleep_ms(getattr(self.cfg, "agent_settle_ms", 1400))
            self.cb.clear()

        self.cb.say("I reached the step limit without finishing.")
        self.cb.failed(f"step cap ({max_steps}) reached")
