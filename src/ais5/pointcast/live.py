"""Live voice sessions: continuous listening with utterance endpointing.

``LiveListener`` streams the microphone and runs a small energy-based voice
activity detector: it calibrates to ambient noise, waits for speech, captures
until a silence gap, then hands the complete utterance (float32 mono, with a
short pre-roll so the first syllable is not clipped) to ``on_utterance``.

The callback fires on the AUDIO thread — the controller bridges it to the Qt
thread via a queued signal. ``feed()`` is the testable core: tests drive it
with synthetic blocks, no audio device needed.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Callable

from ..utils.logging import get_logger

_log = get_logger("pointcast.live")

# Phrases that end a live session. Matched on a normalized transcript.
_END_PHRASES = {
    "done", "i am done", "im done", "that's all", "thats all", "that is all",
    "stop", "stop listening", "goodbye", "bye",
}


def is_end_phrase(text: str) -> bool:
    t = re.sub(r"[^a-z' ]", "", (text or "").lower()).strip()
    return t in _END_PHRASES or t.startswith("thank")


class LiveListener:
    """Energy-endpointed continuous listener.

    States: calibrating -> waiting -> capturing -> (utterance) -> waiting.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        *,
        on_utterance: Callable[[Any], None],
        stream_factory: Callable[..., Any] | None = None,
        block_ms: int = 32,
        start_ms: int = 130,  # this much voiced audio starts a capture
        end_silence_ms: int = 900,  # this much silence ends it
        preroll_ms: int = 260,  # audio kept from just before speech started
        min_speech_ms: int = 240,  # discard captures with less voiced audio (noise)
        max_utterance_s: float = 12.0,
        calib_ms: int = 400,  # ambient-noise calibration window
        threshold_mult: float = 3.5,
        min_threshold: float = 0.006,
        max_threshold: float = 0.05,  # ceiling: noisy calibration can't lock out speech
    ) -> None:
        self.samplerate = samplerate
        self.on_utterance = on_utterance
        self._stream_factory = stream_factory
        self.block = max(1, int(samplerate * block_ms / 1000))
        self._need_start = max(1, start_ms // block_ms)
        self._need_end = max(1, end_silence_ms // block_ms)
        self._preroll = deque(maxlen=max(1, preroll_ms // block_ms))
        self._min_voiced = max(1, min_speech_ms // block_ms)
        self._max_blocks = max(1, int(max_utterance_s * 1000 / block_ms))
        self._calib_need = max(1, calib_ms // block_ms)
        self._mult = threshold_mult
        self._min_thr = min_threshold
        self._max_thr = max_threshold

        self._state = "calibrating"
        self._calibrated = False  # calibrate once per listener, not per resume
        self._calib: list[float] = []
        self._thr = min_threshold
        self._voiced_run = 0
        self._silent_run = 0
        self._voiced_total = 0
        self._frames: list[Any] = []
        self._stream: Any = None

    # ── device lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        if self._stream is not None:
            return
        # Re-opening the mic mid-session keeps the session's noise calibration;
        # only a fresh listener calibrates.
        self._reset("waiting" if self._calibrated else "calibrating")
        self._calib = []

        def _callback(indata, _frames, _time, _status):  # noqa: ANN001 (audio thread)
            try:
                self.feed(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())
            except Exception:  # noqa: BLE001 — the audio thread must never die
                _log.exception("live listener feed crashed")

        if self._stream_factory is not None:
            stream = self._stream_factory(callback=_callback)
        else:
            import sounddevice as sd

            stream = sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="float32",
                blocksize=self.block, callback=_callback,
            )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._stream = stream

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            except Exception:  # noqa: BLE001
                _log.warning("live listener stream stop failed", exc_info=True)
            finally:
                # close() must run even when stop() raises, or the PortAudio
                # handle leaks and the mic can stay busy for the next session.
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
        self._reset("calibrating")

    @property
    def running(self) -> bool:
        return self._stream is not None

    # ── the endpointing state machine (pure; tests drive this directly) ─────
    def _reset(self, state: str) -> None:
        self._state = state
        self._voiced_run = 0
        self._silent_run = 0
        self._voiced_total = 0
        self._frames = []
        self._preroll.clear()

    def _rms(self, block: Any) -> float:
        import numpy as np

        return float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0

    def feed(self, block: Any) -> None:
        rms = self._rms(block)

        if self._state == "calibrating":
            self._calib.append(rms)
            if len(self._calib) >= self._calib_need:
                ambient = sorted(self._calib)[len(self._calib) // 2]  # median
                # Clamp: a noisy window (or a bit of our own TTS tail) must not
                # push the threshold so high that normal speech never triggers.
                self._thr = min(self._max_thr, max(self._min_thr, ambient * self._mult))
                self._state = "waiting"
                self._calibrated = True
                _log.info("live: calibrated (ambient %.4f -> threshold %.4f)", ambient, self._thr)
            return

        voiced = rms >= self._thr

        if self._state == "waiting":
            self._preroll.append(block)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self._need_start:
                self._frames = list(self._preroll)
                self._state = "capturing"
                self._silent_run = 0
                self._voiced_total = self._voiced_run
            return

        # capturing
        self._frames.append(block)
        if voiced:
            self._voiced_total += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
        if self._silent_run >= self._need_end or len(self._frames) >= self._max_blocks:
            enough_speech = self._voiced_total >= self._min_voiced
            if not enough_speech:
                # A door slam or cough: too little voiced audio to be a command.
                _log.info("live: discarding %d-block noise capture", len(self._frames))
                self._reset("waiting")
                return
            import numpy as np

            samples = np.concatenate(self._frames).astype("float32")
            self._reset("waiting")
            self.on_utterance(samples)
