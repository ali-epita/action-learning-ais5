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

import concurrent.futures
import time

from PySide6 import QtCore

from ..utils.logging import get_logger
from .capture import capture_screen
from .clicker import Clicker
from .config import PointCastConfig
from .engine import GroundingEngine
from .locator import confirmation_phrase
from .recorder import MicRecorder
from .speech import Speaker
from .stt import SpeechToText
from .ui import ConfirmHUD, CrosshairOverlay, DisambiguateHUD, InputBar, StatusHUD

CAPTURE_DELAY_MS = 160  # let our windows hide before grabbing the screen
CLICK_DELAY_MS = 80  # let the overlay disappear before clicking
_log = get_logger("pointcast")


class _Bridge(QtCore.QObject):
    """Carries results from the inference thread back to the UI thread (queued)."""

    ground_done = QtCore.Signal(object)  # (GroundResult, CoordinateMapper)
    ground_failed = QtCore.Signal(str)
    stt_done = QtCore.Signal(str)
    stt_failed = QtCore.Signal(str)
    warmed = QtCore.Signal(float)  # seconds


class Controller(QtCore.QObject):
    def __init__(self, cfg: PointCastConfig, engine: GroundingEngine):
        super().__init__()
        self.cfg = cfg
        self.engine = engine
        self.speaker = Speaker(
            cfg.tts_engine, cfg.tts_voice,
            neural_model=cfg.tts_neural_model, neural_voice=cfg.tts_neural_voice,
        )
        self.clicker = Clicker(dry_run=cfg.dry_run)
        self.overlay = CrosshairOverlay(cfg)
        self.hud = ConfirmHUD(cfg)
        self.input_bar = InputBar(cfg)
        self.disambig = DisambiguateHUD(cfg)
        self.status = StatusHUD(cfg)
        self.recorder = MicRecorder(cfg.stt_samplerate) if cfg.enable_voice else None
        self.stt = (
            SpeechToText(cfg.stt_model, cfg.stt_samplerate, cfg.stt_min_seconds)
            if cfg.enable_voice else None
        )

        # Single dedicated thread for ALL MLX work (see module docstring).
        self._infer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pc-infer")
        self.sig = _Bridge()
        self.sig.ground_done.connect(self.on_ground_done)
        self.sig.ground_failed.connect(self.on_ground_failed)
        self.sig.stt_done.connect(self.on_transcribed)
        self.sig.stt_failed.connect(self.on_transcribe_failed)
        self.sig.warmed.connect(lambda s: _log.info("model warm and ready in %.0fs", s))

        self._target = ""
        self._pending: tuple[float, float] | None = None  # logical click point
        self._candidates_logical: list[tuple[float, float]] = []
        self._logical_size: tuple[int, int] = (0, 0)
        self._busy = False
        self._recording = False
        self._record_trigger: str | None = None  # "ptt" | "button" — who started recording

        self.connect_widgets()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def warmup_async(self) -> None:
        """Warm the model on the inference thread (first MLX op also binds the
        GPU stream to that thread, which every later call reuses)."""
        def _job() -> None:
            _log.info("warming model (first call compiles Metal kernels, ~60s)...")
            t0 = time.perf_counter()
            self.engine.warmup()
            self.sig.warmed.emit(time.perf_counter() - t0)

        self._infer.submit(_job)

    def shutdown(self) -> None:
        self._infer.shutdown(wait=False, cancel_futures=True)

    # ── entry ────────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def start_interaction(self) -> None:
        if self._busy:
            _log.info("hotkey ignored (an interaction is already in progress)")
            return
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
        self._target = text
        _log.info("target received: %r", text)
        self.input_bar.dismiss()
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self._busy = True
        QtCore.QTimer.singleShot(CAPTURE_DELAY_MS, self._launch_ground)

    def _launch_ground(self) -> None:
        _log.info("hiding UI, capturing screen, grounding (model working)...")
        self._infer.submit(self._ground_job, self._target)

    def _ground_job(self, target: str) -> None:
        """Runs on the single inference thread."""
        try:
            t0 = time.perf_counter()
            frame = capture_screen(self.cfg.ground_max_side_or_none, self.cfg.monitor_index)
            _log.info(
                "captured screen %dx%d px -> grounding image %dx%d px (scale %.3f)",
                frame.full_image.size[0], frame.full_image.size[1],
                frame.image.size[0], frame.image.size[1], frame.mapper.ground_scale,
            )
            result = self.engine.ground(frame.image, target)
            _log.info("grounding finished in %.1fs", time.perf_counter() - t0)
            self.sig.ground_done.emit((result, frame.mapper))
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.sig.ground_failed.emit(repr(e))

    # ── grounding result ─────────────────────────────────────────────────────
    @QtCore.Slot(object)
    def on_ground_done(self, payload: object) -> None:
        result, mapper = payload  # type: ignore[misc]
        self._logical_size = mapper.logical_size
        if result.point is None:
            _log.info("no match - refusing and asking again")
            self._busy = False
            self.speaker.speak("I couldn't find that. Try describing it differently.")
            self.start_interaction()
            return

        accepted = getattr(result, "accepted", True)
        cands = result.candidates or [result.point]
        if not accepted and len(cands) >= 2:
            cands_logical = [mapper.ground_to_logical(*c) for c in cands][: self.cfg.max_candidates]
            self._disambiguate(cands_logical)
            return

        lx, ly = mapper.ground_to_logical(*result.point)
        self._pending = (lx, ly)
        _log.info(
            "mapped to logical screen point (%.0f, %.0f); badge=%s accepted=%s",
            lx, ly, result.badge, accepted,
        )
        self.overlay.show_point(lx, ly)
        lw, lh = mapper.logical_size
        phrase = confirmation_phrase(self._target, lx, ly, lw, lh)
        # Refuse-and-ask: when the gate didn't lock it, never auto-click.
        mode = self.cfg.confirm_mode if accepted else "explicit"
        if not accepted:
            phrase = f"Not sure - {phrase}"
        _log.info("showing confirmation (mode=%s): %s", mode, phrase)
        if self.cfg.speak_locator:
            self.speaker.speak(phrase)
        self.hud.show_confirm(phrase, result.badge, self.cfg.countdown_seconds, mode)

    @QtCore.Slot(str)
    def on_ground_failed(self, msg: str) -> None:
        self._busy = False
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self.speaker.speak("Something went wrong.")
        _log.error("grounding failed: %s", msg)

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
        if self.cfg.speak_locator:
            self.speaker.speak(phrase)
        self.hud.show_confirm(phrase, "chosen", self.cfg.countdown_seconds, "explicit")

    # ── confirm / cancel / retry ─────────────────────────────────────────────
    @QtCore.Slot()
    def on_confirmed(self) -> None:
        if self._pending is None:
            return
        lx, ly = self._pending
        _log.info("confirmed -> dispatching click")
        self.hud.dismiss()
        self.overlay.clear()
        QtCore.QTimer.singleShot(CLICK_DELAY_MS, lambda: self._do_click(lx, ly))

    def _do_click(self, lx: float, ly: float) -> None:
        # macOS first-mouse fix: a click on a non-key window only activates it, so
        # bring the window under the target to the front first, then click. Skipped
        # in dry-run (never change the real system) and when disabled in config.
        activated = False
        if self.cfg.activate_target_window and not self.cfg.dry_run:
            import os

            from .window_focus import focus_window_at

            activated = focus_window_at(lx, ly, skip_pid=os.getpid())
        delay = self.cfg.activate_settle_ms if activated else 0
        QtCore.QTimer.singleShot(delay, lambda: self._dispatch_click(lx, ly))

    def _dispatch_click(self, lx: float, ly: float) -> None:
        try:
            _log.info("CLICK at logical (%.0f, %.0f)%s", lx, ly, " [dry-run]" if self.cfg.dry_run else "")
            self.clicker.click(lx, ly)
        finally:
            self._pending = None
            self._busy = False
            _log.info("interaction complete")

    @QtCore.Slot()
    def on_retry(self) -> None:
        _log.info("retry requested")
        self.overlay.clear()
        self.hud.dismiss()
        self.disambig.dismiss()
        self._pending = None
        self._busy = False
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
        self._infer.submit(self._stt_job, samples)

    def _stt_job(self, samples) -> None:
        """Runs on the single inference thread (whisper is also MLX)."""
        try:
            self.sig.stt_done.emit(self.stt.transcribe(samples))
        except Exception as e:  # noqa: BLE001
            self.sig.stt_failed.emit(repr(e))

    @QtCore.Slot(str)
    def on_transcribed(self, text: str) -> None:
        self._record_trigger = None
        text = (text or "").strip()
        if not text:
            _log.info("no speech recognized")
            self.input_bar.set_state("idle")
            self.input_bar.set_hint("I didn't catch that — try again, or type it")
            self.speaker.speak("I didn't catch that.")
            return
        _log.info("transcribed: %r", text)
        # Dictate into the search field; the user reviews it and presses Enter to
        # point. (If the bar was dismissed mid-transcription, fall back to grounding.)
        if self.input_bar.isVisible():
            self.input_bar.set_state("idle")
            self.input_bar.set_text(text)
        else:
            self.on_target(text)

    @QtCore.Slot(str)
    def on_transcribe_failed(self, msg: str) -> None:
        self._record_trigger = None
        self.input_bar.set_state("idle")
        _log.error("transcription failed: %s", msg)

    # ── teardown ─────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _abort(self) -> None:
        _log.info("cancelled")
        if self._recording:  # stop any in-flight dictation cleanly
            self._recording = False
            self._record_trigger = None
            try:
                self.recorder.stop()
            except Exception:  # noqa: BLE001
                pass
        self.speaker.stop()
        self.overlay.clear()
        self.hud.dismiss()
        self.input_bar.dismiss()
        self.disambig.dismiss()
        self.status.dismiss()
        self._pending = None
        self._candidates_logical = []
        self._busy = False

    # wiring done after construction (signals from widgets)
    def connect_widgets(self) -> None:
        self.input_bar.submitted.connect(self.on_target)
        self.input_bar.canceled.connect(self._abort)
        self.input_bar.mic_clicked.connect(self.on_mic_clicked)
        self.hud.confirmed.connect(self.on_confirmed)
        self.hud.canceled.connect(self._abort)
        self.hud.retry.connect(self.on_retry)
        self.disambig.picked.connect(self.on_picked)
        self.disambig.canceled.connect(self._abort)
        self.disambig.retry.connect(self.on_retry)
