#!/usr/bin/env python3
"""Diagnose PointCast push-to-talk: isolate microphone vs. key detection.

Runs two independent checks so we know exactly which half is failing:

  STEP 1  microphone + offline STT  -> records 3s, prints level + transcription
  STEP 2  push-to-talk key          -> 12s window, logs every press/release of
                                        the configured key and whether it matches

    .venv-demo/bin/python scripts/pointcast_voice_doctor.py            # tests cmd_r (the default)
    .venv-demo/bin/python scripts/pointcast_voice_doctor.py ctrl_r     # tests a different key
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ais5.pointcast import app as A
from ais5.pointcast.config import PointCastConfig
from ais5.pointcast.stt import SpeechToText
from ais5.utils.logging import get_logger, setup_logging

setup_logging("INFO", None)
log = get_logger("voice-doctor")

KEY = sys.argv[1] if len(sys.argv) > 1 else PointCastConfig().ptt_key
SR = 16000


def step1_mic() -> bool:
    print("\n" + "=" * 60)
    print("STEP 1 — microphone + offline STT")
    print("=" * 60)
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001
        print(f"  sounddevice import failed: {e!r}")
        return False

    try:
        dev = sd.query_devices(kind="input")
        print(f"  default input device: {dev['name']!r} ({int(dev['default_samplerate'])} Hz)")
    except Exception as e:  # noqa: BLE001
        print(f"  no input device / mic permission missing: {e!r}")
        return False

    print('  >> SPEAK NOW for 3 seconds (e.g. say "click the search box") ...')
    try:
        buf = sd.rec(int(3 * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:  # noqa: BLE001
        print(f"  recording failed (grant Microphone permission to your terminal): {e!r}")
        return False

    samples = buf.reshape(-1).astype("float32")
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    print(f"  captured {len(samples)} samples | RMS={rms:.4f} peak={peak:.4f}")
    if peak < 0.005:
        print("  RESULT: mic captured (near) SILENCE — check input device / mic permission / mute.")
        return False

    text = SpeechToText().transcribe(samples)
    print(f'  transcription: "{text}"')
    ok = bool(text.strip())
    print(f"  RESULT: microphone + STT {'OK' if ok else 'captured sound but no transcription'}")
    return ok


def step2_key() -> bool:
    print("\n" + "=" * 60)
    print(f"STEP 2 — push-to-talk key detection (testing: {KEY!r})")
    print("=" * 60)
    from pynput import keyboard

    A._prewarm_macos_accessibility(log)  # avoid the pyobjc race in this process too
    target = A._resolve_key(keyboard, KEY)
    print(f"  resolved {KEY!r} -> {target!r}")
    print("  >> Press and release the key a few times within 12 seconds ...\n")

    hits = {"press": 0, "release": 0, "other": 0}

    def on_press(key):
        if A._key_matches(key, target):
            hits["press"] += 1
            print(f"    MATCH press   {key!r}")
        else:
            hits["other"] += 1

    def on_release(key):
        if A._key_matches(key, target):
            hits["release"] += 1
            print(f"    MATCH release {key!r}")

    lis = keyboard.Listener(on_press=on_press, on_release=on_release)
    lis.daemon = True
    lis.start()
    time.sleep(12)
    lis.stop()

    print(f"\n  matched presses={hits['press']} releases={hits['release']} "
          f"(other keys seen: {hits['other']})")
    if hits["press"] == 0 and hits["other"] == 0:
        print("  RESULT: NO key events at all — grant Accessibility permission to your terminal.")
        return False
    ok = hits["press"] >= 1 and hits["release"] >= 1
    print(f"  RESULT: key {KEY!r} {'detected with clean press+release' if ok else 'NOT cleanly detected'}")
    return ok


def main() -> int:
    print(f"PointCast voice doctor — testing PTT key {KEY!r}")
    mic_ok = step1_mic()
    key_ok = step2_key()
    print("\n" + "=" * 60)
    print(f"SUMMARY:  microphone+STT = {'PASS' if mic_ok else 'FAIL'}   "
          f"key({KEY}) = {'PASS' if key_ok else 'FAIL'}")
    print("=" * 60)
    if mic_ok and key_ok:
        print("Both halves work — push-to-talk should function. Remember to HOLD the")
        print("key while speaking (a quick tap is shorter than the 0.3s minimum).")
    return 0 if (mic_ok and key_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
