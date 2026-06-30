"""Microphone capture for push-to-talk voice input (sounddevice).

Records mono float32 at 16 kHz (whisper's expected rate) into a buffer between
``start()`` and ``stop()``. Requires macOS Microphone permission. A custom
``stream_factory`` can be injected for headless testing.
"""

from __future__ import annotations

from typing import Any, Callable


class MicRecorder:
    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        stream_factory: Callable[..., Any] | None = None,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self._stream_factory = stream_factory
        self._frames: list[Any] = []
        self._stream: Any = None

    def start(self) -> None:
        self._frames = []

        def _callback(indata, _frames, _time, _status):  # noqa: ANN001
            self._frames.append(indata.copy())

        if self._stream_factory is not None:
            self._stream = self._stream_factory(callback=_callback)
        else:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=self.channels, dtype="float32", callback=_callback
            )
        self._stream.start()

    def stop(self):
        import numpy as np

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        if not self._frames:
            return np.zeros(0, dtype="float32")
        audio = np.concatenate(self._frames, axis=0).reshape(-1).astype("float32")
        self._frames = []
        return audio

    @property
    def is_recording(self) -> bool:
        return self._stream is not None
