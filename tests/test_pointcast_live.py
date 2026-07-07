"""Tests for the live-voice endpointer (pure: synthetic blocks, no audio)."""

from __future__ import annotations

import numpy as np

from ais5.pointcast.live import LiveListener, is_end_phrase


def _mk(on_utterance, **kw):
    defaults = dict(
        samplerate=16000, block_ms=32, start_ms=96, end_silence_ms=160,
        preroll_ms=64, calib_ms=96, min_threshold=0.01, max_utterance_s=2.0,
        min_speech_ms=96,
    )
    defaults.update(kw)
    return LiveListener(on_utterance=on_utterance, **defaults)


def _blocks(listener, level, n):
    return [np.full(listener.block, level, dtype=np.float32) for _ in range(n)]


def test_calibrates_then_detects_one_utterance():
    got = []
    lis = _mk(got.append)
    for b in _blocks(lis, 0.001, 3):  # calibration (3 blocks of near-silence)
        lis.feed(b)
    for b in _blocks(lis, 0.001, 2):  # ambient
        lis.feed(b)
    for b in _blocks(lis, 0.2, 5):  # speech (>= 3 voiced blocks to start)
        lis.feed(b)
    for b in _blocks(lis, 0.001, 6):  # silence gap (>= 5 blocks ends it)
        lis.feed(b)
    assert len(got) == 1
    # captured speech plus pre-roll, concatenated float32
    assert got[0].dtype == np.float32 and len(got[0]) >= 5 * lis.block


def test_short_blip_does_not_trigger():
    got = []
    lis = _mk(got.append)
    for b in _blocks(lis, 0.001, 3):
        lis.feed(b)
    lis.feed(_blocks(lis, 0.3, 1)[0])  # one voiced block only (below start_ms)
    for b in _blocks(lis, 0.001, 8):
        lis.feed(b)
    assert got == []


def test_max_utterance_cutoff():
    got = []
    lis = _mk(got.append, max_utterance_s=0.2)  # ~6 blocks at 32ms
    for b in _blocks(lis, 0.001, 3):
        lis.feed(b)
    for b in _blocks(lis, 0.2, 40):  # non-stop speech
        lis.feed(b)
    assert len(got) >= 1  # forced cut instead of capturing forever


def test_multiple_utterances_in_sequence():
    got = []
    lis = _mk(got.append)
    for b in _blocks(lis, 0.001, 3):
        lis.feed(b)
    for _ in range(2):
        for b in _blocks(lis, 0.2, 4):
            lis.feed(b)
        for b in _blocks(lis, 0.001, 6):
            lis.feed(b)
    assert len(got) == 2


def test_noise_burst_below_min_speech_is_discarded():
    """A door slam crosses the threshold briefly: enough to start a capture but
    with too little voiced audio to be a command — it must not be emitted."""
    got = []
    lis = _mk(got.append, min_speech_ms=192)  # needs 6 voiced blocks
    for b in _blocks(lis, 0.001, 3):
        lis.feed(b)
    for b in _blocks(lis, 0.3, 3):  # 3 voiced blocks: starts capture, too short
        lis.feed(b)
    for b in _blocks(lis, 0.001, 6):
        lis.feed(b)
    assert got == []


def test_calibration_is_kept_across_restarts():
    got = []
    lis = _mk(got.append)
    for b in _blocks(lis, 0.001, 3):  # calibrate once
        lis.feed(b)
    assert lis._state == "waiting" and lis._calibrated
    lis._reset("waiting")
    # simulate a stop/start cycle without a device: start() path chooses state
    lis._state = "waiting" if lis._calibrated else "calibrating"
    for b in _blocks(lis, 0.2, 4):
        lis.feed(b)
    for b in _blocks(lis, 0.001, 6):
        lis.feed(b)
    assert len(got) == 1  # no re-calibration ate the speech


def test_end_phrases():
    for t in ("done", "Done.", "thank you", "Thanks!", "thank you very much",
              "stop listening", "that's all", "goodbye"):
        assert is_end_phrase(t), t
    for t in ("read my grade", "click done button", "type thanks into the box", ""):
        assert not is_end_phrase(t), t


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} live tests passed")
