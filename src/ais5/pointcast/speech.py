"""Text-to-speech for the spoken confirmation locator + countdown.

Two engines, switchable:
  - "say":    macOS built-in (``/usr/bin/say``) — instant, zero model load,
              fully offline, rock-solid for the live confirm path. Default.
  - "neural": mlx_audio on-device neural voice (Kokoro) — nicer voice for the
              showcase; lazy-loaded and played on a worker thread, with an
              automatic fall back to ``say`` if anything goes wrong.

All non-blocking by default so speech never stalls the UI thread.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any

from ..utils.logging import get_logger

_log = get_logger("pointcast.tts")


class Speaker:
    def __init__(
        self,
        engine: str = "say",
        voice: str | None = None,
        neural_model: str = "prince-canuma/Kokoro-82M",
        neural_voice: str = "af_heart",
    ):
        self.engine = engine
        self.voice = voice
        self.neural_model = neural_model
        self.neural_voice = neural_voice
        self._proc: subprocess.Popen | None = None
        self._neural: Any = None

    def speak(self, text: str, *, blocking: bool = False) -> None:
        if not text or self.engine == "off":
            return
        if self.engine == "neural":
            self._speak_neural(text, blocking)
        else:
            self._speak_say(text, blocking)

    # ── macOS native ─────────────────────────────────────────────────────────
    def _speak_say(self, text: str, blocking: bool) -> None:
        say = shutil.which("say") or "/usr/bin/say"
        args = [say]
        if self.voice:
            args += ["-v", self.voice]
        args.append(text)
        self.stop()
        try:
            if blocking:
                subprocess.run(args, check=False)
            else:
                self._proc = subprocess.Popen(args)
        except Exception:
            pass  # speech is best-effort; never crash the pointer over TTS

    # ── neural (mlx_audio) ───────────────────────────────────────────────────
    def _load_neural(self) -> None:
        if self._neural is None:
            from mlx_audio.tts import load as load_tts

            _log.info("loading neural TTS model %s ...", self.neural_model)
            self._neural = load_tts(self.neural_model)

    def _speak_neural(self, text: str, blocking: bool) -> None:
        try:
            self._load_neural()
        except Exception as e:  # noqa: BLE001
            _log.warning("neural TTS unavailable (%r); falling back to say", e)
            self._speak_say(text, blocking)
            return

        def _run() -> None:
            try:
                from mlx_audio.tts.generate import generate_audio

                generate_audio(
                    text, model=self._neural, voice=self.neural_voice,
                    play=True, verbose=False, save=False,
                )
            except Exception as e:  # noqa: BLE001
                _log.warning("neural TTS failed (%r); falling back to say", e)
                self._speak_say(text, blocking=True)

        if blocking:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None
