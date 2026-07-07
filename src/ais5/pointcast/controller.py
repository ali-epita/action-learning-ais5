"""PointCast interaction controller — the state machine that ties everything
together: hotkey -> input bar -> capture -> ground -> crosshair (or numbered
candidates) -> spoken locator + countdown -> real OS click, or cancel / retry /
disambiguate.

THREADING: MLX GPU streams are thread-bound — calling MLX from more than one
thread (or concurrently) crashes ("no Stream(gpu) in current thread" / segfault).
So ALL model work (warmup, grounding, transcription) runs on a single dedicated
inference thread (a 1-worker executor), serially. Results come back to the UI
thread via queued Qt signals. Our own windows are hidden before the screen is
captured and before the click is dispatched. Every step is logged.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from PySide6 import QtCore

from ..utils.logging import get_logger
from .capture import capture_screen
from .clicker import Clicker
from .config import PointCastConfig
from .engine import GroundingEngine
from .live import LiveListener, is_end_phrase
from .locator import confirmation_phrase
from .recorder import MicRecorder
from .speech import Speaker
from .stt import SpeechToText
from .ui import (
    ConfirmHUD,
    CrosshairOverlay,
    DisambiguateHUD,
    InputBar,
    LiveHUD,
    RecordHUD,
    ResultCard,
    SettingsPanel,
    StatusHUD,
    TrustPanel,
)

CAPTURE_DELAY_MS = 160  # let our windows hide before grabbing the screen
CLICK_DELAY_MS = 80  # let the overlay disappear before clicking

# Spoken/shown names for normalized key names ("esc" reads badly out loud).
_KEY_LABELS = {"esc": "Escape", "ctrl": "Control", "backspace": "Delete",
               "pageup": "Page Up", "pagedown": "Page Down"}
# Typed/spoken phrases that start a live session (exact match, conservative).
_LIVE_START_PHRASES = {"start listening", "go live", "live mode", "live session", "live", "start live"}
_log = get_logger("pointcast")


class _Bridge(QtCore.QObject):
    """Carries results from the inference thread back to the UI thread (queued)."""

    ground_done = QtCore.Signal(object)  # (generation, GroundResult, CoordinateMapper)
    ground_failed = QtCore.Signal(object)  # (generation, message)
    stt_done = QtCore.Signal(object)  # (generation, text)
    stt_failed = QtCore.Signal(str)
    read_done = QtCore.Signal(object)  # (generation, query, answer)
    read_failed = QtCore.Signal(object)  # (generation, message)
    live_utterance = QtCore.Signal(object)  # raw samples from the live listener (audio thread)
    live_text = QtCore.Signal(object)  # (generation, transcript) for a live command
    # Emitted right AFTER the screenshot is taken (so a status pill shown in
    # response can never leak into the frame) and before the model call, so
    # the UI can show a live "thinking" indicator during the seconds of MLX.
    captured = QtCore.Signal(object)  # (generation, kind: "ground" | "read")
    warmed = QtCore.Signal(object)  # (ok, seconds, error_message)
    # multi-step task mode (emitted from the inference thread, handled on the UI thread)
    task_status = QtCore.Signal(str)
    task_say = QtCore.Signal(str)
    task_point = QtCore.Signal(object)  # (lx, ly)
    task_clear = QtCore.Signal()
    task_hide = QtCore.Signal(object)  # hide all overlays before a screenshot; payload = ack Event
    task_answer = QtCore.Signal(str)
    task_failed = QtCore.Signal(str)
    task_done = QtCore.Signal()
    # trust panel (M5)
    trust_mem = QtCore.Signal(object)  # peak-memory bytes | None
    trust_hash = QtCore.Signal(str)


class _InferenceWorker:
    """Single DAEMON thread that runs all MLX work serially.

    MLX GPU streams are thread-bound, so every model call (warmup, grounding,
    STT, neural TTS) must run on this one thread. A daemon thread (unlike
    ThreadPoolExecutor's non-daemon workers) means interpreter exit never
    blocks on an in-flight Metal kernel compile: quitting during the ~60s
    warmup exits immediately instead of hanging until the compile finishes.
    ``shutdown()`` drains queued jobs so nothing new runs after quit.
    """

    _STOP = object()

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name="pc-infer", daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> None:
        if not self._closed:
            self._q.put((fn, args))

    def shutdown(self) -> None:
        self._closed = True
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(self._STOP)

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job is self._STOP or self._closed:
                return
            fn, args = job
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — jobs handle their own errors; this is the backstop
                _log.exception("inference job crashed")


class Controller(QtCore.QObject):
    def __init__(self, cfg: PointCastConfig, engine: GroundingEngine):
        super().__init__()
        self.cfg = cfg
        self.engine = engine
        # Single dedicated thread for ALL MLX work (see module docstring).
        # Created first so the Speaker can serialize neural TTS onto it.
        self._infer = _InferenceWorker()
        self.speaker = Speaker(
            cfg.tts_engine, cfg.tts_voice,
            neural_model=cfg.tts_neural_model, neural_voice=cfg.tts_neural_voice,
            mlx_submit=self._infer.submit,
        )
        self.clicker = Clicker(dry_run=cfg.dry_run)
        self.overlay = CrosshairOverlay(cfg)
        self.hud = ConfirmHUD(cfg)
        self.input_bar = InputBar(cfg)
        self.disambig = DisambiguateHUD(cfg)
        self.status = StatusHUD(cfg)
        self.record_hud = RecordHUD(cfg)  # Save/Cancel controls while recording
        self.result_card = ResultCard(cfg)  # spoken read-back answer, shown large
        self.settings_panel = SettingsPanel(cfg)  # runtime toggles (gear button)
        self.live_hud = LiveHUD(cfg)  # the live-session surface (transcript + state)
        self.trust: TrustPanel | None = TrustPanel(cfg) if cfg.show_trust_panel else None
        self.recorder = MicRecorder(cfg.stt_samplerate) if cfg.enable_voice else None
        self.stt = (
            SpeechToText(cfg.stt_model, cfg.stt_samplerate, cfg.stt_min_seconds)
            if cfg.enable_voice else None
        )

        self.sig = _Bridge()
        self.sig.ground_done.connect(self.on_ground_done)
        self.sig.ground_failed.connect(self.on_ground_failed)
        self.sig.stt_done.connect(self.on_transcribed)
        self.sig.stt_failed.connect(self.on_transcribe_failed)
        self.sig.read_done.connect(self.on_read_done)
        self.sig.read_failed.connect(self.on_read_failed)
        self.sig.live_utterance.connect(self.on_live_utterance)
        self.sig.live_text.connect(self.on_live_text)
        self.sig.captured.connect(self.on_captured)
        self.sig.warmed.connect(self.on_warmed)
        self.sig.task_status.connect(self.on_task_status)
        self.sig.task_say.connect(self.on_task_say)
        self.sig.task_point.connect(self.on_task_point)
        self.sig.task_clear.connect(self.on_task_clear)
        self.sig.task_hide.connect(self.on_task_hide)
        self.sig.task_answer.connect(self.on_task_answer)
        self.sig.task_failed.connect(self.on_task_failed)
        self.sig.task_done.connect(self.on_task_done)
        if self.trust is not None:
            self.sig.trust_mem.connect(self.trust.set_memory)
            self.sig.trust_hash.connect(self.trust.set_hash)
            self._start_trust_panel()

        self._target = ""
        self._pending: tuple[float, float] | None = None  # logical click point
        self._pending_action: dict[str, Any] = {"kind": "click"}  # what to do on confirm
        self._read_gen = 0  # generation counter for read requests
        self._live_active = False  # a hands-free live voice session is running
        self._live_processing = False  # an utterance is being transcribed/acted
        self._live_listener: LiveListener | None = None
        self._candidates_logical: list[tuple[float, float]] = []
        self._logical_size: tuple[int, int] = (0, 0)
        self._busy = False
        self._recording = False
        self._record_trigger: str | None = None  # "ptt" | "button" — who started recording
        # Generation counters: _abort() bumps them, so results from jobs that
        # were already in flight when the user cancelled are dropped instead of
        # resurrecting as autonomous grounding/clicks.
        self._ground_gen = 0
        self._stt_gen = 0
        self._act_gen = 0  # invalidates queued click/type timers on cancel
        self._task_runner: Any = None  # active multi-step TaskRunner, if any
        self._last_agent_run: dict[str, Any] | None = None  # last SUCCESSFUL agent run (goal + steps)
        self._demo: dict[str, Any] | None = None  # active record-by-demonstration session
        self._demo_step_pending = False  # a demo step was appended but not yet clicked
        self._status_timer = QtCore.QTimer(self)  # delayed status dismiss (single, cancelable)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status.dismiss())
        self._recipes: tuple = ()  # built-in + user recipes (loaded when task mode is on)
        if cfg.enable_tasks or cfg.enable_agent:
            # Agent mode needs the recipe store too: a successful agent run can
            # be saved as a recipe and replayed by name afterwards.
            from .task import all_recipes

            self._recipes = all_recipes(cfg.recipes_path)

        self.connect_widgets()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def warmup_async(self) -> None:
        """Warm the model on the inference thread (first MLX op also binds the
        GPU stream to that thread, which every later call reuses). When voice is
        enabled, the Whisper model is preloaded right after, on the same thread,
        so the first dictation does not stall for its multi-second load."""
        def _job() -> None:
            _log.info("warming model (first call compiles Metal kernels, ~60s)...")
            t0 = time.perf_counter()
            ok, err = self.engine.warmup()
            self.sig.warmed.emit((ok, time.perf_counter() - t0, err))

        def _stt_preload() -> None:
            try:
                self.stt.load()
            except Exception as e:  # noqa: BLE001
                _log.warning("STT preload failed (voice will retry lazily): %r", e)

        self._infer.submit(_job)
        if self.stt is not None:
            self._infer.submit(_stt_preload)

    @QtCore.Slot(object)
    def on_warmed(self, payload: object) -> None:
        ok, seconds, err = payload  # type: ignore[misc]
        if ok:
            _log.info("model warm and ready in %.0fs", seconds)
            return
        _log.error("model failed to load: %s", err)
        self._show_status("Model failed to load — see the log")
        self.speaker.speak("The model failed to load. Check the log.")

    def shutdown(self) -> None:
        # Order matters: stop the active task FIRST so no further OS clicks can
        # dispatch after the user quits, then stop speech, then drain the
        # inference queue (daemon thread: an in-flight Metal compile cannot
        # block process exit), then release backend temp files.
        if self._task_runner is not None:
            self._task_runner.abort()
        try:
            self.speaker.stop()
        except Exception:  # noqa: BLE001
            pass
        self._infer.shutdown()
        try:
            self.engine.backend.close()
        except Exception:  # noqa: BLE001
            pass

    # ── entry ────────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def start_interaction(self) -> None:
        if self._live_active:
            # During a live voice session the hotkey is the exit: end the
            # session and cancel whatever is in flight.
            _log.info("hotkey during live session: ending it")
            self._end_live_session(quiet=True)
            if self._task_runner is not None:
                self._task_runner.abort()
            elif self._busy:
                self._abort()
            self.speaker.speak("Done.")
            return
        if self._busy:
            if self._task_runner is not None:  # hotkey during a task = cancel it
                _log.info("hotkey during a task: aborting")
                self._task_runner.abort()
                return
            # The hotkey is the universal escape: it works even when the confirm
            # HUD failed to take key focus (macOS can deny activation to a Tool
            # window), so a countdown can always be cancelled.
            _log.info("hotkey during an interaction: cancelling it")
            self._abort()
        _log.info("interaction start: showing input bar")
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.input_bar.prompt(self._target)

    # ── target submitted ─────────────────────────────────────────────────────
    @QtCore.Slot(str)
    def on_target(self, text: str) -> None:
        if self._recording:
            return  # ignore Enter while a dictation is in progress
        if self._busy:
            _log.info("target %r ignored (an interaction is already in progress)", text)
            return
        _log.info("target received: %r", text)
        low = text.lower().strip().rstrip(".")
        # Enter a live session by phrase (hands-free entry), gated on the
        # capability and checked before any other routing so it can't be a recipe.
        if self.cfg.enable_live_voice and not self._live_active and self._demo is None:
            if low in _LIVE_START_PHRASES:
                _log.info("live session requested by phrase: %r", low)
                self._start_live_session()
                return
        if (
            self.cfg.enable_agent and self._demo is None
            and low.startswith("do ") and len(low.split()) >= 2
        ):
            goal = text.strip()[3:].strip()
            _log.info("agent goal: %r", goal)
            self._start_agent(goal)
            return
        # record-by-demonstration commands (record / save / cancel) take priority.
        if self.cfg.enable_tasks or self.cfg.enable_agent:
            from .task import parse_command as parse_record_command

            cmd = parse_record_command(text)
            if cmd is not None:
                self._handle_record_command(*cmd)
                return

        # While demonstrating, every phrase is a click step (no verb routing).
        if self._demo is not None:
            self._demo["steps"].append(text)
            self._demo_step_pending = True
            _log.info("recorded step %d: %r", len(self._demo["steps"]), text)
            self._pending_action = {"kind": "click"}
            self._begin_capture(text)
            return

        # Verb routing: read / type / click (click is the default).
        from .command import parse_command as parse_verb

        intent = parse_verb(text)
        if intent.kind == "read":
            self._start_read(intent.query)
            return
        if intent.kind == "type":
            self._start_type(intent.text, intent.target, intent.submit)
            return
        if intent.kind == "press":
            self._start_press(intent)
            return

        # click: an explicit "click ..." is literal; a bare phrase may be a recipe.
        target = intent.target
        if not target:
            # A bare verb ("click", on its own) has nothing to ground.
            self.speaker.speak("Tell me what to click.")
            self._reprompt()
            return
        if not intent.explicit_click and (self.cfg.enable_tasks or self.cfg.enable_agent):
            from .task import match_recipe

            recipe = match_recipe(target, self._recipes)
            if recipe is not None:
                _log.info("matched task recipe: %s", recipe.name)
                self._start_task(recipe)
                return
        self._pending_action = {"kind": "click", "button": intent.button}
        self._begin_capture(target)

    def _dismiss_all_surfaces(self) -> None:
        """Hide every PointCast surface: before a screenshot (nothing of ours may
        leak into the frame the model grounds on) and before a confirm HUD."""
        self.input_bar.dismiss()
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.status.dismiss()  # the "Recording: ..." banner must not be in the screenshot
        self.record_hud.dismiss()
        self.settings_panel.dismiss()  # nor the Save/Cancel controls
        self.result_card.dismiss()  # nor a lingering read card
        self.live_hud.dismiss()  # nor the live session surface (re-shown on resume)
        self._trust_visible(False)  # nor the trust panel

    def _begin_capture(self, target: str) -> None:
        """Hide all overlays and kick off a screenshot + grounding for ``target``."""
        self._target = target
        self._dismiss_all_surfaces()
        self._busy = True
        # Bind the generation NOW: if the user cancels during the capture delay,
        # _abort() bumps _ground_gen and this launch must stay invalidated
        # instead of re-reading the fresh counter and resurrecting the request.
        gen = self._ground_gen
        QtCore.QTimer.singleShot(CAPTURE_DELAY_MS, lambda: self._launch_ground(gen))

    def _launch_ground(self, gen: int) -> None:
        if gen != self._ground_gen:
            _log.info("grounding launch discarded (cancelled during capture delay)")
            return
        _log.info("hiding UI, capturing screen, grounding (model working)...")
        self._infer.submit(self._ground_job, self._target, gen)

    def _ground_job(self, target: str, gen: int) -> None:
        """Runs on the single inference thread."""
        try:
            t0 = time.perf_counter()
            frame = capture_screen(self.cfg.ground_max_side_or_none, self.cfg.monitor_index)
            _log.info(
                "captured screen %dx%d px -> grounding image %dx%d px (scale %.3f)",
                frame.full_image.size[0], frame.full_image.size[1],
                frame.image.size[0], frame.image.size[1], frame.mapper.ground_scale,
            )
            self.sig.captured.emit((gen, "ground"))  # safe: frame already taken
            result = self.engine.ground(
                frame.image, target,
                full_image=frame.full_image, full_scale=frame.mapper.ground_scale,
            )
            _log.info("grounding finished in %.1fs", time.perf_counter() - t0)
            self.sig.ground_done.emit((gen, result, frame.mapper))
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.sig.ground_failed.emit((gen, repr(e)))

    def _no_match(self) -> None:
        """The model produced nothing clickable (no answer, a degenerate corner
        point, or a point off the screen): refuse and ask again."""
        _log.info("no match - refusing and asking again")
        self._pop_pending_demo_step("no match")
        self._busy = False
        self.speaker.speak("I couldn't find that. Try describing it differently.")
        if self._live_active:
            self._maybe_resume_live()  # stay hands-free: just listen again
        else:
            self.start_interaction()

    def _pop_pending_demo_step(self, reason: str) -> None:
        """Drop the last recorded demo step when its interaction did not end in
        a dispatched click (no match, error, retry, or cancel), so saved recipes
        never contain steps the user rejected."""
        if self._demo is not None and self._demo_step_pending and self._demo["steps"]:
            dropped = self._demo["steps"].pop()
            _log.info("dropped unrecorded step %r (%s)", dropped, reason)
        self._demo_step_pending = False

    # ── grounding result ─────────────────────────────────────────────────────
    @QtCore.Slot(object)
    def on_ground_done(self, payload: object) -> None:
        gen, result, mapper = payload  # type: ignore[misc]
        if gen != self._ground_gen:
            _log.info("stale grounding result discarded (cancelled)")
            return
        self.status.dismiss()  # the thinking pill
        self._logical_size = mapper.logical_size
        from .geometry import is_degenerate_point, validate_logical_point

        if result.point is not None and is_degenerate_point(result.point):
            # (0,0)-ish is the model's "not found", not a target.
            _log.info("degenerate point %s -> treated as no match", result.point)
            result.point = None
        if result.point is None:
            self._no_match()
            return

        accepted = getattr(result, "accepted", True)
        cands = result.candidates or [result.point]
        if not accepted and len(cands) >= 2:
            cands_logical = [
                pt for c in cands
                if not is_degenerate_point(c)
                and (pt := validate_logical_point(*mapper.ground_to_logical(*c), mapper.logical_size))
                is not None
            ][: self.cfg.max_candidates]
            if len(cands_logical) >= 2:
                self._disambiguate(cands_logical)
                return
            if not cands_logical:
                _log.info("all candidates off-screen/degenerate -> treated as no match")
                self._no_match()
                return
            # a single plausible candidate left: fall through to the normal
            # path below (accepted is False, so it always confirms explicitly)

        lx, ly = mapper.ground_to_logical(*result.point)
        checked = validate_logical_point(lx, ly, mapper.logical_size)
        if checked is None:
            # The model answered outside the screen: that is a failed grounding,
            # not a click. Never let it reach the mouse.
            _log.info("point (%.0f, %.0f) is outside the %s screen -> treated as no match",
                      lx, ly, mapper.logical_size)
            self._no_match()
            return
        lx, ly = checked
        self._pending = (lx, ly)
        _log.info(
            "mapped to logical screen point (%.0f, %.0f); badge=%s accepted=%s",
            lx, ly, result.badge, accepted,
        )
        self.overlay.show_point(lx, ly)
        lw, lh = mapper.logical_size
        # Refuse-and-ask: when the gate didn't lock it, never auto-click.
        mode = self.cfg.confirm_mode if accepted else "explicit"
        if self._pending_action.get("kind") == "type":
            typed = self._pending_action.get("text", "")
            phrase = f"Type “{typed}” into {self._target}"
            badge = "type"
        else:
            phrase = confirmation_phrase(self._target, lx, ly, lw, lh)
            badge = result.badge
            if self._pending_action.get("button") == "right":
                phrase = phrase.replace("Clicking", "Right-clicking", 1)
                badge = "right click"
        if not accepted:
            phrase = f"Not sure - {phrase}"
        _log.info("showing confirmation (mode=%s): %s", mode, phrase)
        if self.cfg.speak_locator:
            self.speaker.speak(phrase)
        self.hud.show_confirm(phrase, badge, self.cfg.countdown_seconds, mode)

    @QtCore.Slot(object)
    def on_ground_failed(self, payload: object) -> None:
        gen, msg = payload  # type: ignore[misc]
        if gen != self._ground_gen:
            _log.info("stale grounding failure discarded (cancelled): %s", msg)
            return
        self.status.dismiss()  # the thinking pill
        self._pop_pending_demo_step("grounding error")
        self._busy = False
        self._trust_visible(True)
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.speaker.speak("Something went wrong.")
        _log.error("grounding failed: %s", msg)
        self._maybe_resume_live()

    # ── disambiguation (M3) ──────────────────────────────────────────────────
    def _disambiguate(self, cands_logical: list[tuple[float, float]]) -> None:
        self._candidates_logical = cands_logical
        _log.info("uncertain: presenting %d candidates for disambiguation", len(cands_logical))
        for i, (x, y) in enumerate(cands_logical, start=1):
            _log.info("    candidate %d at (%.0f, %.0f)", i, x, y)
        self.overlay.show_markers_only(cands_logical)
        self.hud.dismiss()
        self.speaker.speak("I'm not sure. Did you mean one of these? Press the number.")
        self.disambig.show_options(len(cands_logical))

    @QtCore.Slot(int)
    def on_picked(self, idx: int) -> None:
        if not self._candidates_logical or idx >= len(self._candidates_logical):
            return
        lx, ly = self._candidates_logical[idx]
        _log.info("user picked candidate %d at (%.0f, %.0f)", idx + 1, lx, ly)
        self._pending = (lx, ly)
        self.disambig.dismiss()
        self.overlay.show_point(lx, ly)
        lw, lh = self._logical_size
        phrase = confirmation_phrase(self._target, lx, ly, lw, lh)
        if self._pending_action.get("button") == "right":
            phrase = phrase.replace("Clicking", "Right-clicking", 1)
        if self.cfg.speak_locator:
            self.speaker.speak(phrase)
        self.hud.show_confirm(phrase, "chosen", self.cfg.countdown_seconds, "explicit")

    # ── confirm / cancel / retry ─────────────────────────────────────────────
    @QtCore.Slot()
    def on_confirmed(self) -> None:
        action = self._pending_action
        # The token every queued timer in the click/type chain carries; _abort()
        # bumps it, so a cancel during CLICK_DELAY_MS or the activation settle
        # still stops the dispatch. Consuming the pending state here also makes
        # confirmation one-shot: a double Enter (or countdown + Enter racing)
        # cannot queue the same click or typing twice.
        gen = self._act_gen
        # Key press: no grounded point involved, dispatch after the usual delay.
        if action.get("kind") == "press":
            self._pending_action = {"kind": "click"}
            _log.info("confirmed -> pressing keys")
            self.hud.dismiss()
            self.overlay.clear()
            QtCore.QTimer.singleShot(CLICK_DELAY_MS, lambda: self._finish_press(action, gen))
            return
        # Type into the currently-focused field (no target was grounded).
        if action.get("kind") == "type" and action.get("no_target"):
            self._pending_action = {"kind": "click"}
            _log.info("confirmed -> typing into the focused field")
            self.hud.dismiss()
            self.overlay.clear()
            QtCore.QTimer.singleShot(CLICK_DELAY_MS, lambda: self._finish_type(action, gen))
            return
        if self._pending is None:
            return
        lx, ly = self._pending
        self._pending = None
        self.hud.dismiss()
        self.overlay.clear()
        if action.get("kind") == "type":
            _log.info("confirmed -> click field then type")
            QtCore.QTimer.singleShot(CLICK_DELAY_MS, lambda: self._do_type(lx, ly, action, gen))
        else:
            _log.info("confirmed -> dispatching click")
            button = action.get("button", "left")
            QtCore.QTimer.singleShot(CLICK_DELAY_MS, lambda: self._do_click(lx, ly, gen, button))

    def _act_cancelled(self, gen: int, what: str) -> bool:
        if gen != self._act_gen:
            _log.info("queued %s discarded (cancelled)", what)
            return True
        return False

    def _do_click(self, lx: float, ly: float, gen: int, button: str = "left") -> None:
        if self._act_cancelled(gen, "click"):
            return
        # macOS first-mouse fix: a click on a non-key window only activates it, so
        # bring the window under the target to the front first, then click. Skipped
        # in dry-run (never change the real system) and when disabled in config.
        activated = False
        if self.cfg.activate_target_window and not self.cfg.dry_run:
            import os

            from .window_focus import focus_window_at

            activated = focus_window_at(lx, ly, skip_pid=os.getpid())
        delay = self.cfg.activate_settle_ms if activated else 0
        QtCore.QTimer.singleShot(delay, lambda: self._dispatch_click(lx, ly, gen, button))

    def _dispatch_click(self, lx: float, ly: float, gen: int, button: str = "left") -> None:
        if self._act_cancelled(gen, "click"):
            return
        try:
            _log.info("%s at logical (%.0f, %.0f)%s", "RIGHT-CLICK" if button == "right" else "CLICK",
                      lx, ly, " [dry-run]" if self.cfg.dry_run else "")
            if button == "right":
                self.clicker.right_click(lx, ly)
            else:
                self.clicker.click(lx, ly)
        finally:
            self._pending = None
            self._busy = False
            self._trust_visible(True)
            if self._demo is not None:
                self._demo_step_pending = False  # the step was actually clicked: keep it
                self._show_record_hud()
            self._maybe_resume_live()
            _log.info("interaction complete")

    # ── type action ──────────────────────────────────────────────────────────
    def _start_type(self, text: str, target: str, submit: bool) -> None:
        if not text:
            self.speaker.speak("Tell me what to type.")
            self._reprompt()
            return
        self._pending_action = {"kind": "type", "text": text, "submit": submit, "no_target": not target}
        if target:
            self._begin_capture(target)  # ground the field, then confirm + type
            return
        # No field named: type into whatever is focused, after a confirm.
        self._dismiss_all_surfaces()
        self._busy = True
        phrase = f"Type “{text}” into the focused field"
        if self.cfg.speak_locator:
            self.speaker.speak(f"Type {text}")
        self.hud.show_confirm(phrase, "type", self.cfg.countdown_seconds, self.cfg.confirm_mode)

    def _do_type(self, lx: float, ly: float, action: dict, gen: int) -> None:
        if self._act_cancelled(gen, "type"):
            return
        # Focus the field first (first-mouse fix), click it, then type.
        activated = False
        if self.cfg.activate_target_window and not self.cfg.dry_run:
            import os

            from .window_focus import focus_window_at

            activated = focus_window_at(lx, ly, skip_pid=os.getpid())
        delay = self.cfg.activate_settle_ms if activated else 0
        QtCore.QTimer.singleShot(delay, lambda: self._click_then_type(lx, ly, action, gen))

    def _click_then_type(self, lx: float, ly: float, action: dict, gen: int) -> None:
        if self._act_cancelled(gen, "type"):
            return
        _log.info("CLICK field at (%.0f, %.0f) then type", lx, ly)
        self.clicker.click(lx, ly)
        # let the field take focus before typing
        QtCore.QTimer.singleShot(180, lambda: self._finish_type(action, gen))

    def _finish_type(self, action: dict, gen: int) -> None:
        if self._act_cancelled(gen, "typing"):
            return
        try:
            text = action.get("text", "")
            _log.info("TYPE %r%s%s", text, " + enter" if action.get("submit") else "",
                      " [dry-run]" if self.cfg.dry_run else "")
            self.clicker.type_text(text)
            if action.get("submit"):
                self.clicker.press_enter()
        finally:
            self._pending = None
            self._pending_action = {"kind": "click"}
            self._busy = False
            self._trust_visible(True)
            self._maybe_resume_live()
            _log.info("type complete")

    # ── press action ─────────────────────────────────────────────────────────
    def _start_press(self, intent) -> None:  # noqa: ANN001 (CommandIntent)
        if not intent.keys:
            # Unknown key names are refused, never guessed: a wrong key press
            # can submit a form or close a window.
            if intent.text:
                self.speaker.speak(f"I don't know the key {intent.text}. Try again.")
            else:
                self.speaker.speak("Tell me what to press.")
            self._reprompt()
            return
        self._pending_action = {"kind": "press", "keys": tuple(intent.keys)}
        self._dismiss_all_surfaces()
        self._busy = True
        phrase = "Press " + " + ".join(_KEY_LABELS.get(k, k.capitalize()) for k in intent.keys)
        _log.info("press request: %s", "+".join(intent.keys))
        if self.cfg.speak_locator:
            self.speaker.speak(phrase)
        self.hud.show_confirm(phrase, "press", self.cfg.countdown_seconds, self.cfg.confirm_mode)

    def _finish_press(self, action: dict, gen: int) -> None:
        if self._act_cancelled(gen, "key press"):
            return
        try:
            keys = tuple(action.get("keys", ()))
            _log.info("PRESS %s%s", "+".join(keys), " [dry-run]" if self.cfg.dry_run else "")
            self.clicker.press_keys(keys)
        finally:
            self._pending = None
            self._busy = False
            self._trust_visible(True)
            self._maybe_resume_live()
            _log.info("press complete")

    # ── read action ──────────────────────────────────────────────────────────
    def _start_read(self, query: str) -> None:
        if not query:
            self.speaker.speak("What should I read?")
            self._reprompt()
            return
        self._read_gen += 1
        gen = self._read_gen
        self._dismiss_all_surfaces()
        self._busy = True
        _log.info("read request: %r", query)
        QtCore.QTimer.singleShot(CAPTURE_DELAY_MS, lambda: self._infer.submit(self._read_job, query, gen))

    def _read_job(self, query: str, gen: int) -> None:
        """Runs on the single inference thread: capture + VQA."""
        from .command import read_prompt

        try:
            frame = capture_screen(self.cfg.ground_max_side_or_none, self.cfg.monitor_index)
            self.sig.captured.emit((gen, "read"))  # safe: frame already taken
            answer = self.engine.ask(frame.image, read_prompt(query))
            self.sig.read_done.emit((gen, query, (answer or "").strip()))
        except Exception as e:  # noqa: BLE001
            self.sig.read_failed.emit((gen, repr(e)))

    @QtCore.Slot(object)
    def on_captured(self, payload: object) -> None:
        """Show the thinking pill the moment the screenshot is in hand: the
        model is now working and the pill can no longer leak into the frame."""
        gen, kind = payload  # type: ignore[misc]
        if kind == "read":
            if gen != self._read_gen:
                return
            self.status.show_busy("Reading the screen", "#a78bfa")
            return
        if gen != self._ground_gen or self._task_runner is not None:
            return  # stale, or task mode (which narrates its own steps)
        if self._pending_action.get("kind") == "type":
            self.status.show_busy(f"Finding {self._target}", "#34d399")
        else:
            self.status.show_busy(f"Locating {self._target}", "#60a5fa")

    @QtCore.Slot(object)
    def on_read_done(self, payload: object) -> None:
        gen, query, answer = payload  # type: ignore[misc]
        if gen != self._read_gen:
            _log.info("stale read result discarded (cancelled)")
            return
        self.status.dismiss()  # the thinking pill
        self._busy = False
        self._trust_visible(True)
        if not answer:
            self.speaker.speak("I couldn't read that.")
            self._maybe_resume_live()
            return
        _log.info("read answer: %s", answer)
        self.result_card.show_answer(query, answer)
        self.speaker.speak(answer)
        QtCore.QTimer.singleShot(9000, self.result_card.dismiss)
        self._maybe_resume_live()

    @QtCore.Slot(object)
    def on_read_failed(self, payload: object) -> None:
        gen, msg = payload  # type: ignore[misc]
        if gen != self._read_gen:
            return
        self.status.dismiss()  # the thinking pill
        self._busy = False
        self._trust_visible(True)
        self.speaker.speak("Something went wrong reading the screen.")
        _log.error("read failed: %s", msg)
        self._maybe_resume_live()

    @QtCore.Slot()
    def on_retry(self) -> None:
        _log.info("retry requested")
        self._pop_pending_demo_step("retry")
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self._pending = None
        self._pending_action = {"kind": "click"}
        self._busy = False
        if self._live_active:
            self.speaker.speak("Okay, tell me again.")
            self._maybe_resume_live()
        else:
            self.start_interaction()

    # ── voice input (M4): only while the search bar is open ───────────────────
    # The user must press the hotkey first; voice (push-to-talk key OR the mic
    # button in the bar) then dictates into the search field.
    @QtCore.Slot()
    def on_mic_clicked(self) -> None:
        """Mic button in the bar: click to start, click again to stop (toggle)."""
        if self._recording:
            self._stop_listening()
        else:
            self._start_listening("button")

    @QtCore.Slot()
    def on_ptt_press(self) -> None:
        """Push-to-talk key down — ignored unless the search bar is open."""
        if self.recorder is None or self._busy or self._recording:
            return
        if not self.input_bar.isVisible():
            return  # require the hotkey (search bar) first
        self._start_listening("ptt")

    @QtCore.Slot()
    def on_ptt_release(self) -> None:
        """Push-to-talk key up — only stops a recording the PTT key started."""
        if self._recording and self._record_trigger == "ptt":
            self._stop_listening()

    def _start_listening(self, trigger: str) -> None:
        if self.recorder is None or self._recording:
            return
        # A new recording invalidates any previous transcription still in
        # flight: its late result must not overwrite this recording's text.
        self._stt_gen += 1
        try:
            self.recorder.start()
        except Exception as e:  # noqa: BLE001
            _log.warning("microphone unavailable: %r", e)
            self.input_bar.set_state("idle")
            self.input_bar.set_hint("Microphone unavailable — enable it in System Settings")
            try:
                self.speaker.speak("Microphone unavailable. Grant microphone permission.")
            except Exception:  # noqa: BLE001
                pass
            return
        self._recording = True
        self._record_trigger = trigger
        _log.info("listening (%s)...", trigger)
        self.input_bar.set_state("listening")

    def _stop_listening(self) -> None:
        if not self._recording:
            return
        self._recording = False
        try:
            samples = self.recorder.stop()
        except Exception as e:  # noqa: BLE001
            _log.warning("recorder stop failed: %r", e)
            self.input_bar.set_state("idle")
            return
        _log.info("transcribing %d samples...", len(samples))
        self.input_bar.set_state("transcribing")
        self._infer.submit(self._stt_job, samples, self._stt_gen)

    def _stt_job(self, samples, gen: int) -> None:
        """Runs on the single inference thread (whisper is also MLX)."""
        try:
            self.sig.stt_done.emit((gen, self.stt.transcribe(samples)))
        except Exception as e:  # noqa: BLE001
            self.sig.stt_failed.emit(repr(e))

    @QtCore.Slot(object)
    def on_transcribed(self, payload: object) -> None:
        gen, text = payload  # type: ignore[misc]
        self._record_trigger = None
        if gen != self._stt_gen:
            # The user cancelled (Esc) while this transcription was in flight;
            # acting on it now would be an unrequested grounding or task run.
            _log.info("stale dictation discarded (cancelled): %r", (text or "")[:60])
            return
        text = (text or "").strip()
        if not text:
            _log.info("no speech recognized")
            self.input_bar.set_state("idle")
            self.input_bar.set_hint("I didn't catch that — try again, or type it")
            self.speaker.speak("I didn't catch that.")
            return
        _log.info("transcribed: %r", text)
        # Dictate into the search field; the user reviews it and presses Enter to
        # point. Never act autonomously on speech when the bar is gone.
        if self.input_bar.isVisible():
            self.input_bar.set_state("idle")
            self.input_bar.set_text(text)
        else:
            _log.info("input bar no longer visible; dictation %r not executed", text)

    @QtCore.Slot(str)
    def on_transcribe_failed(self, msg: str) -> None:
        self._record_trigger = None
        self.input_bar.set_state("idle")
        _log.error("transcription failed: %s", msg)

    # ── record-by-demonstration ──────────────────────────────────────────────
    @staticmethod
    def _readback_question(readback: str) -> str:
        """Turn a plain-language read-back ("the battery percentage") into a
        VQA prompt the model answers off the final screen."""
        return (
            f"Look at this screenshot and tell the user {readback}. "
            "Answer in one short sentence with the value."
        )

    def _handle_record_command(self, kind: str, name: str | None) -> None:
        if kind == "save_as":
            # Distill the last successful agent run into a named recipe.
            self.input_bar.dismiss()
            run = self._last_agent_run
            if not run:
                self.speaker.speak("There is no completed task to save yet.")
                return
            from .task import Recipe, Step, all_recipes, save_user_recipe

            # "name :: readback" packs an optional spoken result into the name.
            name, _, readback = (name or "").partition(" :: ")
            name = name.strip()
            question = self._readback_question(readback.strip()) if readback.strip() else ""
            # The recipe answers to its name AND to the original goal phrase,
            # so repeating the goal (without "do") replays the fast path.
            utterances = tuple(dict.fromkeys((name.lower(), run["goal"].lower())))
            recipe = Recipe(name=name, utterances=utterances,
                            steps=tuple(Step(target=t) for t in run["steps"]),
                            question=question)
            try:
                save_user_recipe(recipe, self.cfg.recipes_path)
                self._recipes = all_recipes(self.cfg.recipes_path)
                self._last_agent_run = None
                _log.info("agent run saved as recipe %r (%d steps, readback=%s)",
                          name, len(recipe.steps), bool(question))
                self.speaker.speak(
                    f"Saved {name} with {len(recipe.steps)} steps. Just say it to run it."
                )
                self._show_status(f"Saved recipe: {name}")
            except Exception as e:  # noqa: BLE001
                _log.error("could not save agent recipe: %r", e)
                self.speaker.speak("I could not save that recipe.")
            return
        if kind == "record":
            self._demo = {"name": name, "steps": []}
            self._demo_step_pending = False
            self.input_bar.dismiss()
            _log.info("recording recipe %r", name)
            self.speaker.speak(f"Recording {name}. Show me each step, then save.")
            self._show_record_hud()
            return
        if kind == "cancel":
            self._demo = None
            self.record_hud.dismiss()
            self.status.dismiss()
            self.speaker.speak("Recording cancelled.")
            return
        # kind == "save"  (name carries an optional read-back question here)
        readback = name
        demo = self._demo
        self._demo = None
        self.record_hud.dismiss()
        self.settings_panel.dismiss()
        if not demo or not demo["steps"]:
            self.status.dismiss()
            self.speaker.speak("Nothing to save yet.")
            return
        from .task import Recipe, Step, all_recipes, save_user_recipe

        question = self._readback_question(readback) if readback else ""
        recipe = Recipe(name=demo["name"], utterances=(demo["name"].lower(),),
                        steps=tuple(Step(target=t) for t in demo["steps"]),
                        question=question)
        try:
            save_user_recipe(recipe, self.cfg.recipes_path)
            self._recipes = all_recipes(self.cfg.recipes_path)  # make it matchable now
            spoken = " and it will read the result" if question else ""
            self.speaker.speak(f"Saved {recipe.name} with {len(recipe.steps)} steps{spoken}.")
            self._show_status(f"Saved: {recipe.name}")
        except Exception as e:  # noqa: BLE001
            _log.error("could not save recipe: %r", e)
            self.speaker.speak("I could not save that recipe.")

    # ── multi-step task mode (Stage 1) ───────────────────────────────────────
    def _task_callbacks(self):
        from .task import TaskCallbacks

        return TaskCallbacks(
            status=self.sig.task_status.emit,
            say=self.sig.task_say.emit,
            point=lambda x, y: self.sig.task_point.emit((x, y)),
            clear=self.sig.task_clear.emit,
            hide=self.sig.task_hide.emit,
            answer=self.sig.task_answer.emit,
            failed=self.sig.task_failed.emit,
            done=self.sig.task_done.emit,
        )

    def _start_task(self, recipe) -> None:
        self.input_bar.dismiss()
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.live_hud.dismiss()
        self._busy = True
        from .task import TaskRunner
        from .window_focus import focus_window_at

        self._task_runner = TaskRunner(
            self.cfg, self.engine, self.clicker, self._task_callbacks(),
            focus_fn=(None if self.cfg.dry_run else focus_window_at),
        )
        _log.info("starting task: %s", recipe.name)
        self._infer.submit(self._task_runner.run, recipe)

    def _start_agent(self, goal: str) -> None:
        """Free-form agent mode: same wiring, abort path, and busy state as a
        recipe task — only the step source differs (model-planned)."""
        self.input_bar.dismiss()
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.live_hud.dismiss()
        self._busy = True
        from .task.agent import AgentRunner
        from .window_focus import focus_window_at

        self._task_runner = AgentRunner(
            self.cfg, self.engine, self.clicker, self._task_callbacks(),
            focus_fn=(None if self.cfg.dry_run else focus_window_at),
        )
        self._infer.submit(self._task_runner.run, goal)

    def _show_status(self, text: str) -> None:
        """Show the status banner and cancel any pending delayed dismiss, so a
        stale timer from a previous task can never hide a fresh banner."""
        self._status_timer.stop()
        self.status.show_status(text)

    # ── in-app settings (gear button) ────────────────────────────────────────
    @QtCore.Slot()
    def on_settings_clicked(self) -> None:
        if self.settings_panel.isVisible():
            self.settings_panel.dismiss()
        else:
            self.settings_panel.open_near(self.input_bar)

    @QtCore.Slot()
    def on_live_clicked(self) -> None:
        """The ◉ live button: start a hands-free session from the command bar."""
        if not self._live_active:
            self._start_live_session()

    def _reprompt(self) -> None:
        """Ask again after an empty/unusable command: stay hands-free in a live
        session, otherwise reopen the command bar."""
        if self._live_active:
            self._maybe_resume_live()
        else:
            self.start_interaction()

    def _ensure_voice(self) -> None:
        """Create the recorder/STT lazily when voice is enabled from settings
        (they are only built at startup when --voice was passed)."""
        if self.recorder is None:
            self.recorder = MicRecorder(self.cfg.stt_samplerate)
        if self.stt is None:
            self.stt = SpeechToText(self.cfg.stt_model, self.cfg.stt_samplerate, self.cfg.stt_min_seconds)

            def _preload() -> None:
                try:
                    self.stt.load()
                except Exception as e:  # noqa: BLE001
                    _log.warning("STT preload failed (voice will retry lazily): %r", e)

            self._infer.submit(_preload)

    @QtCore.Slot(str, object)
    def on_setting_changed(self, key: str, value: object) -> None:
        _log.info("setting %s -> %r", key, value)
        setattr(self.cfg, key, value)
        if key == "enable_voice":
            if value:
                self._ensure_voice()
            elif self._recording:  # turning voice off mid-dictation
                self._stt_gen += 1
                self._recording = False
                try:
                    self.recorder.stop()
                except Exception:  # noqa: BLE001
                    pass
            self.input_bar.set_voice_enabled(bool(value))
        elif key == "enable_live_voice":
            self.input_bar.set_live_enabled(bool(value))
            if not value and self._live_active:  # turned off mid-session
                self._end_live_session(quiet=True)
        elif key in ("enable_tasks", "enable_agent") and value and not self._recipes:
            from .task import all_recipes

            self._recipes = all_recipes(self.cfg.recipes_path)
        elif key == "show_trust_panel":
            self._set_trust_panel(bool(value))
        elif key == "dry_run":
            self.clicker.dry_run = bool(value)
        elif key == "tts_engine":
            self.speaker.engine = str(value)
        # Persist so the panel remembers across launches.
        from .settings import ALL_FIELDS, save_settings

        try:
            save_settings({k: getattr(self.cfg, k) for k in ALL_FIELDS}, self.cfg.settings_path)
        except Exception as e:  # noqa: BLE001
            _log.warning("could not persist settings: %r", e)

    # ── live voice sessions ──────────────────────────────────────────────────
    def _start_live_session(self) -> None:
        """Hands-free loop: listen -> transcribe -> act -> listen again, until
        an end phrase ("done", "thank you", "stop listening") or the hotkey.
        A fresh listener each session so ambient calibration is current."""
        self._ensure_voice()
        self._live_listener = LiveListener(
            samplerate=self.cfg.stt_samplerate,
            on_utterance=self.sig.live_utterance.emit,
            end_silence_ms=self.cfg.live_end_silence_ms,
        )
        self._live_active = True
        self._live_processing = False
        self._stt_gen += 1  # a fresh generation for this session's utterances
        _log.info("live session started")
        self.input_bar.dismiss()
        self.settings_panel.dismiss()
        self.live_hud.begin()
        self.speaker.speak("Listening.")
        self._maybe_resume_live()

    def _end_live_session(self, *, quiet: bool = False) -> None:
        if not self._live_active:
            return
        self._live_active = False
        self._live_processing = False
        self._stt_gen += 1  # drop any in-flight live transcript from this session
        if self._live_listener is not None:
            self._live_listener.stop()
            self._live_listener = None
        self.status.dismiss()
        self.live_hud.dismiss()
        _log.info("live session ended")
        if not quiet:
            self.speaker.speak("Done.")

    def _maybe_resume_live(self) -> None:
        """Reopen the mic when the live session is idle. Waits for our own TTS
        to finish (plus a settle) so the calibration/first capture never
        transcribes PointCast's own voice."""
        if not self._live_active or self._live_listener is None or self._busy:
            return
        if self.speaker.is_speaking():
            QtCore.QTimer.singleShot(200, self._maybe_resume_live)
            return
        self._live_processing = False
        # A short settle after TTS finishes clears the `say` tail (and covers
        # neural TTS, whose completion is_speaking() cannot track).
        QtCore.QTimer.singleShot(300, self._open_live_mic)

    def _open_live_mic(self) -> None:
        if not self._live_active or self._live_listener is None or self._busy:
            return
        if self.speaker.is_speaking():  # TTS restarted in the meantime
            QtCore.QTimer.singleShot(200, self._maybe_resume_live)
            return
        try:
            self._live_listener.start()
        except Exception as e:  # noqa: BLE001
            _log.warning("live mic unavailable: %r", e)
            self._end_live_session(quiet=True)
            self.speaker.speak("Microphone unavailable. Grant microphone permission.")
            self.start_interaction()  # fall back to the command bar
            return
        self.live_hud.set_state("listening")
        self.live_hud.set_transcript("")

    @QtCore.Slot(object)
    def on_live_utterance(self, samples: object) -> None:
        # One utterance at a time: ignore anything that races in before we have
        # closed the mic and handled the current transcript.
        if not self._live_active or self._busy or self._live_processing:
            return
        self._live_processing = True
        self._live_listener.stop()
        self.live_hud.set_state("thinking", "transcribing")
        self._infer.submit(self._live_stt_job, samples, self._stt_gen)

    def _live_stt_job(self, samples, gen: int) -> None:
        """Runs on the single inference thread."""
        try:
            self.sig.live_text.emit((gen, self.stt.transcribe(samples)))
        except Exception as e:  # noqa: BLE001
            _log.warning("live transcription failed: %r", e)
            self.sig.live_text.emit((gen, ""))

    @QtCore.Slot(object)
    def on_live_text(self, payload: object) -> None:
        gen, text = payload  # type: ignore[misc]
        if not self._live_active or gen != self._stt_gen or self._busy:
            return  # stale (session changed) or an action is already running
        text = (text or "").strip().rstrip(".")
        if len(text) < 2:
            _log.info("live: nothing usable heard; listening again")
            self._maybe_resume_live()
            return
        _log.info("live command: %r", text)
        self.live_hud.set_transcript(text)  # show the user what was heard
        if is_end_phrase(text):
            self.live_hud.set_state("acting", "ending")
            self._end_live_session()
            return
        self.live_hud.set_state("acting")
        self.live_hud.add_turn(text)  # always record what was said
        self.on_target(text)
        # Commands that finish synchronously (e.g. "record X", "save recipe")
        # never set _busy; reopen the mic right away for the next one.
        if not self._busy and self._task_runner is None:
            self._maybe_resume_live()

    def _set_trust_panel(self, on: bool) -> None:
        if on:
            if self.trust is None:
                self.trust = TrustPanel(self.cfg)
                self.sig.trust_mem.connect(self.trust.set_memory)
                self.sig.trust_hash.connect(self.trust.set_hash)
            self._start_trust_panel()
        else:
            timer = getattr(self, "_trust_timer", None)
            if timer is not None:
                timer.stop()
            if self.trust is not None:
                self.trust.dismiss()

    def _show_record_hud(self) -> None:
        """Show the Save/Cancel recording controls for the active demo."""
        if self._demo is not None:
            self.record_hud.show_recording(self._demo["name"], len(self._demo["steps"]))

    @QtCore.Slot()
    def on_record_save_clicked(self) -> None:
        # The button is the pointer path; voice/typed "save recipe [and read …]"
        # still works and can attach a read-back. The button saves as-is.
        self.record_hud.dismiss()
        self.settings_panel.dismiss()
        self._handle_record_command("save", None)

    @QtCore.Slot()
    def on_record_cancel_clicked(self) -> None:
        self.record_hud.dismiss()
        self.settings_panel.dismiss()
        self._handle_record_command("cancel", None)

    # ── trust panel (M5) ─────────────────────────────────────────────────────
    def _start_trust_panel(self) -> None:
        """Live peak-memory readout (polled via the inference thread so MLX is
        only ever touched there) + a one-time weights sha256 on a plain worker
        thread, verified against the provenance manifest when present."""
        existing = getattr(self, "_trust_timer", None)
        if existing is not None:  # re-enabled from settings: just resume
            self.trust.show_panel()
            existing.start()
            return
        backend = getattr(self.engine, "backend", None)
        self.trust.set_model_line(f"{getattr(backend, 'name', 'model')}  ·  {self.cfg.model_path.rsplit('/', 1)[-1]}")
        self.trust.show_panel()

        self._trust_poll_pending = False

        def _poll_job() -> None:
            try:
                b = backend.peak_memory_bytes() if backend is not None else None
            except Exception:  # noqa: BLE001
                b = None
            self.sig.trust_mem.emit(b)
            self._trust_poll_pending = False

        def _tick() -> None:
            if not self._trust_poll_pending:
                self._trust_poll_pending = True
                self._infer.submit(_poll_job)

        self._trust_timer = QtCore.QTimer(self)
        self._trust_timer.setInterval(2000)
        self._trust_timer.timeout.connect(_tick)
        self._trust_timer.start()

        model_path = self.cfg.model_path

        def _hash_job() -> None:
            import glob
            import hashlib
            from pathlib import Path

            files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
            if not files:
                self.sig.trust_hash.emit("weights file not found")
                return
            h = hashlib.sha256()
            with open(files[0], "rb") as f:
                for chunk in iter(lambda: f.read(1 << 23), b""):
                    h.update(chunk)
            digest = h.hexdigest()
            verified = ""
            manifests = (
                Path("checkpoints/qwen2.5-vl-3b-lora-r64/_provenance/artifact_sha256.txt"),
                Path("DEMO_MODELS.sha256"),  # stock showcase models (see PROVENANCE.md)
            )
            for manifest in manifests:
                try:
                    if manifest.exists() and digest in manifest.read_text():
                        verified = "  ✓"
                        break
                except OSError:
                    pass
            self.sig.trust_hash.emit(f"{digest[:16]}…{verified}")

        threading.Thread(target=_hash_job, name="pc-trust-hash", daemon=True).start()

    def _trust_visible(self, visible: bool) -> None:
        if self.trust is None or not self.cfg.show_trust_panel:
            return
        if visible:
            self.trust.show_panel()
        else:
            self.trust.dismiss()

    @QtCore.Slot(str)
    def on_task_status(self, text: str) -> None:
        _log.info("task: %s", text)
        self._show_status(text)

    @QtCore.Slot(str)
    def on_task_say(self, text: str) -> None:
        if self.cfg.speak_locator:
            self.speaker.speak(text)

    @QtCore.Slot(object)
    def on_task_point(self, p: object) -> None:
        lx, ly = p  # type: ignore[misc]
        self.overlay.show_point(lx, ly)

    @QtCore.Slot()
    def on_task_clear(self) -> None:
        self.overlay.clear()

    @QtCore.Slot(object)
    def on_task_hide(self, ack: object = None) -> None:
        # hide every PointCast window so the next task screenshot sees only the
        # real screen, not our status banner or crosshair (which the model would
        # otherwise try to click)
        self.status.dismiss()
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.input_bar.dismiss()
        self.record_hud.dismiss()
        self.settings_panel.dismiss()
        self.result_card.dismiss()
        self.live_hud.dismiss()
        self._trust_visible(False)
        if ack is not None:
            # Acknowledge on the NEXT event-loop turn so the hides have been
            # handed to the window server before the runner screenshots.
            QtCore.QTimer.singleShot(50, ack.set)

    @QtCore.Slot(str)
    def on_task_answer(self, text: str) -> None:
        _log.info("task answer: %s", text)
        self.overlay.clear()
        self._show_status(text)
        self.speaker.speak(text)

    @QtCore.Slot(str)
    def on_task_failed(self, msg: str) -> None:
        _log.error("task failed: %s", msg)
        self.overlay.clear()

    @QtCore.Slot()
    def on_task_done(self) -> None:
        runner = self._task_runner
        self._task_runner = None
        self._busy = False
        self._trust_visible(True)
        self.overlay.clear()
        self._status_timer.start(6000)  # let the answer linger, then clear (cancelable)
        # A successful agent run can be distilled into a recipe: keep its record
        # and offer the save (only when every clicked step has a usable
        # description to re-ground from next time).
        goal = getattr(runner, "goal", "")
        targets = list(getattr(runner, "clicked_targets", ()))
        if getattr(runner, "succeeded", False) and targets and all(targets):
            self._last_agent_run = {"goal": goal, "steps": targets}
            _log.info("agent run save-able: %r (%d steps)", goal, len(targets))
            self.speaker.speak("Done. Say save recipe as, and a name, to remember this.")
        self._maybe_resume_live()

    # ── teardown ─────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _abort(self) -> None:
        _log.info("cancelled")
        # Invalidate any in-flight grounding/dictation/read so their late
        # results are dropped instead of resurrecting as autonomous actions.
        self._ground_gen += 1
        self._stt_gen += 1
        self._read_gen += 1
        self._act_gen += 1  # kill any click/type already queued on a timer
        self._pending_action = {"kind": "click"}
        if self._recording:  # stop any in-flight dictation cleanly
            self._recording = False
            self._record_trigger = None
            try:
                self.recorder.stop()
            except Exception:  # noqa: BLE001
                pass
        self._pop_pending_demo_step("cancelled")
        self.speaker.stop()
        self.overlay.clear()
        self.hud.dismiss()
        self.input_bar.dismiss()
        self.disambig.dismiss()
        self.status.dismiss()
        self.result_card.dismiss()
        self._pending = None
        self._candidates_logical = []
        self._busy = False
        self._trust_visible(True)
        self._end_live_session(quiet=True)
        if self._demo is not None:  # Esc cancels one step, not the whole recording
            self._show_record_hud()

    # wiring done after construction (signals from widgets)
    def connect_widgets(self) -> None:
        self.input_bar.submitted.connect(self.on_target)
        self.input_bar.canceled.connect(self._abort)
        self.input_bar.mic_clicked.connect(self.on_mic_clicked)
        self.input_bar.settings_clicked.connect(self.on_settings_clicked)
        self.input_bar.live_clicked.connect(self.on_live_clicked)
        self.settings_panel.changed.connect(self.on_setting_changed)
        self.hud.confirmed.connect(self.on_confirmed)
        self.hud.canceled.connect(self._abort)
        self.hud.retry.connect(self.on_retry)
        self.disambig.picked.connect(self.on_picked)
        self.disambig.canceled.connect(self._abort)
        self.disambig.retry.connect(self.on_retry)
        self.record_hud.save.connect(self.on_record_save_clicked)
        self.record_hud.cancel.connect(self.on_record_cancel_clicked)
