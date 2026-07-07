"""Microphone capture for push-to-talk voice input (sounddevice).

Records mono float32 at 16 kHz (whisper's expected rate) into a buffer between
``start()`` and ``stop()``. Requires macOS Microphone permission. A custom
``stream_factory`` can be injected for headless testing.
"""

from __future__ import annotations

from typing import Any, Callable

from ..utils.logging import get_logger

_log = get_logger("pointcast.recorder")


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
        self._overflows = 0

    def start(self) -> None:
        if self._stream is not None:
            # Guard against double-start: silently replacing the stream would
            # leak the previous PortAudio handle (and its callback).
            self.stop()
        self._frames = []
        self._overflows = 0

        def _callback(indata, _frames, _time, status):  # noqa: ANN001
            if status:  # input overflow etc. — audio was lost in this block
                self._overflows += 1
            self._frames.append(indata.copy())

        if self._stream_factory is not None:
            stream = self._stream_factory(callback=_callback)
        else:
            import sounddevice as sd

            stream = sd.InputStream(
                samplerate=self.samplerate, channels=self.channels, dtype="float32", callback=_callback
            )
        try:
            stream.start()
        except Exception:
            try:  # do not leak a created-but-unstartable PortAudio stream
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._stream = stream

    def stop(self):
        import numpy as np

        if self._stream is not None:
            s, self._stream = self._stream, None
            try:
                s.stop()
            finally:
                # close() must run even when stop() raises — dropping the
                # reference without closing leaks the PortAudio handle and can
                # leave the microphone busy for the next recording.
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
        if self._overflows:
            # Dropped audio blocks mean the transcript may miss words. Surface
            # it in the log so a mis-heard command is diagnosable; the caller
            # still gets the audio (a partial phrase usually just fails to
            # ground rather than clicking something wrong).
            _log.warning("recording had %d overflowed audio blocks; words may be missing", self._overflows)
        if not self._frames:
            return np.zeros(0, dtype="float32")
        audio = np.concatenate(self._frames, axis=0).reshape(-1).astype("float32")
        self._frames = []
        return audio

    @property
    def is_recording(self) -> bool:
        return self._stream is not None
