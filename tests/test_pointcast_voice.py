"""CPU-only tests for PointCast voice plumbing (no mic, no audio model)."""

from __future__ import annotations

import types

import numpy as np

from ais5.pointcast.recorder import MicRecorder
from ais5.pointcast.stt import _extract_text


class FakeStream:
    def __init__(self, callback, chunks):
        self._cb = callback
        self._chunks = chunks

    def start(self):
        for c in self._chunks:
            self._cb(c, len(c), None, None)

    def stop(self):
        pass

    def close(self):
        pass


def test_recorder_concatenates_frames():
    chunks = [np.ones((100, 1), dtype="float32"), np.ones((150, 1), dtype="float32")]
    rec = MicRecorder(stream_factory=lambda callback: FakeStream(callback, chunks))
    rec.start()
    audio = rec.stop()
    assert audio.shape == (250,)
    assert audio.dtype == np.float32
    assert rec.stop().shape == (0,)  # nothing buffered after stop


def test_stt_extract_text_variants():
    assert _extract_text(types.SimpleNamespace(text="hello")) == "hello"
    assert _extract_text([{"text": "a"}, {"text": "b"}]) == "a b"
    assert _extract_text([types.SimpleNamespace(text="x"), types.SimpleNamespace(text="y")]) == "x y"
    assert _extract_text({"text": "z"}) == "z"
    assert _extract_text(None) == ""


def test_stt_ignores_short_audio():
    from ais5.pointcast.stt import SpeechToText

    stt = SpeechToText(samplerate=16000, min_seconds=0.3)
    # 0.1s < 0.3s threshold -> returns "" without loading any model
    assert stt.transcribe(np.zeros(1600, dtype="float32")) == ""
    assert stt._model is None  # never loaded


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} voice tests passed")
