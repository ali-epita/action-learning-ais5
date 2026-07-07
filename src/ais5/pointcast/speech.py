"""Text-to-speech for the spoken confirmation locator + countdown.

Two engines, switchable:
  - "say":    macOS built-in (``/usr/bin/say``) — instant, zero model load,
              fully offline, rock-solid for the live confirm path. Default.
  - "neural": mlx_audio on-device neural voice (Kokoro) — nicer voice for the
              showcase, with an automatic fall back to ``say``.

THREADING: Kokoro is an MLX model, and MLX GPU streams are thread-bound (see
controller.py) — running it on an ad-hoc thread concurrently with grounding
crashes Metal. So when the controller provides ``mlx_submit`` (its single
inference thread), ALL neural work (lazy load + generate + play) is serialized
there. The trade-off is that a spoken phrase and a grounding call queue behind
each other; ``say`` (the default) has no such constraint. Speech never blocks
the UI thread and is always best-effort.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any, Callable

from ..utils.logging import get_logger

_log = get_logger("pointcast.tts")


class Speaker:
    def __init__(
        self,
        engine: str = "say",
        voice: str | None = None,
        neural_model: str = "prince-canuma/Kokoro-82M",
        neural_voice: str = "af_heart",
        mlx_submit: Callable[..., None] | None = None,
    ):
        self.engine = engine
        self.voice = voice
        self.neural_model = neural_model
        self.neural_voice = neural_voice
        self._mlx_submit = mlx_submit  # controller's single MLX inference thread
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()  # _proc is touched from several threads
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
                with self._proc_lock:
                    self._proc = subprocess.Popen(args)
        except Exception:  # noqa: BLE001
            # Speech is best-effort; never crash the pointer over TTS — but a
            # silently mute accessibility app is hard to diagnose, so log it.
            _log.warning("say TTS failed", exc_info=True)

    # ── neural (mlx_audio) ───────────────────────────────────────────────────
    def _load_neural(self) -> None:
        if self._neural is None:
            from mlx_audio.tts import load as load_tts

            _log.info("loading neural TTS model %s ...", self.neural_model)
            self._neural = load_tts(self.neural_model)

    def _speak_neural(self, text: str, blocking: bool) -> None:
        def _job() -> None:
            try:
                self._load_neural()
                from mlx_audio.tts.generate import generate_audio

                generate_audio(
                    text, model=self._neural, voice=self.neural_voice,
                    play=True, verbose=False, save=False,
                )
            except Exception as e:  # noqa: BLE001
                _log.warning("neural TTS failed (%r); falling back to say", e)
                self._speak_say(text, blocking=True)

        if self._mlx_submit is not None:
            # Serialize with grounding/STT on the single MLX thread; the model
            # is loaded there too, never on the Qt UI thread.
            self._mlx_submit(_job)
        elif blocking:
            _job()
        else:
            # No MLX executor provided (headless/tests): keep the old behavior
            # but on one thread at a time is the caller's responsibility.
            threading.Thread(target=_job, daemon=True).start()

    def is_speaking(self) -> bool:
        """True while the `say` process is talking (best-effort; neural TTS on
        the worker thread is not tracked). Used by live voice sessions so the
        mic never transcribes PointCast's own speech."""
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None
