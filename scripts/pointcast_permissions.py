#!/usr/bin/env python3
"""Check the three macOS permissions PointCast needs, and say what to fix.

macOS attaches these permissions to the APP THAT LAUNCHES PYTHON (your terminal
or IDE), not to Python itself, so they all appear under that app's name in
System Settings. This script reports the launching app and the live status of:

  Accessibility    - move/click the mouse + the global hotkey and PTT key
  Screen Recording - capture the screen for grounding
  Microphone       - voice (push-to-talk) input

    .venv-demo/bin/python scripts/pointcast_permissions.py           # check
    .venv-demo/bin/python scripts/pointcast_permissions.py --open    # also open the Settings panes
    .venv-demo/bin/python scripts/pointcast_permissions.py --prompt  # trigger the macOS grant prompts
"""
from __future__ import annotations

import os
import subprocess
import sys

PANES = {
    "Accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "Screen Recording": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "Microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}

_APP_NAMES = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm",
    "vscode": "Visual Studio Code (Code)",
    "Hyper": "Hyper",
    "WarpTerminal": "Warp",
    "Tabby": "Tabby",
    "ghostty": "Ghostty",
}


def launching_app() -> str:
    tp = os.environ.get("TERM_PROGRAM", "")
    name = _APP_NAMES.get(tp)
    if name:
        return name
    bundle = os.environ.get("__CFBundleIdentifier", "")
    return name or tp or bundle or "your terminal / IDE"


def check_accessibility() -> tuple[str, str]:
    try:
        import HIServices

        return ("GRANTED" if HIServices.AXIsProcessTrusted() else "DENIED",
                "move/click the mouse + global hotkey and PTT key")
    except Exception as e:  # noqa: BLE001
        return ("UNKNOWN", f"could not query ({e!r})")


def check_screen_recording(prompt: bool) -> tuple[str, str]:
    try:
        import Quartz

        if prompt and hasattr(Quartz, "CGRequestScreenCaptureAccess"):
            Quartz.CGRequestScreenCaptureAccess()
        ok = Quartz.CGPreflightScreenCaptureAccess()
        return ("GRANTED" if ok else "DENIED",
                "capture the screen for grounding (after granting, QUIT + relaunch the terminal)")
    except Exception as e:  # noqa: BLE001
        return ("UNKNOWN", f"could not query ({e!r})")


def check_microphone(prompt: bool) -> tuple[str, str]:
    # No AVFoundation here, so probe the device: a denied mic still opens but
    # yields pure silence, so we can only *confirm* it works (peak > 0), not
    # prove denial. Speaking during the voice doctor is the definitive test.
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001
        return ("UNKNOWN", f"sounddevice unavailable ({e!r})")
    try:
        sd.query_devices(kind="input")
    except Exception as e:  # noqa: BLE001
        return ("NO INPUT DEVICE", f"{e!r}")
    try:
        buf = sd.rec(int(0.6 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    except Exception as e:  # noqa: BLE001
        return ("DENIED?", f"stream failed - likely no permission ({e!r})")
    if peak > 0.002:
        return ("GRANTED", f"mic is capturing audio (ambient peak={peak:.4f})")
    return ("UNCONFIRMED", f"stream opened but silent (peak={peak:.4f}); "
                           "run the voice doctor and SPEAK to confirm")


def main() -> int:
    do_open = "--open" in sys.argv
    do_prompt = "--prompt" in sys.argv
    app = launching_app()

    print("PointCast macOS permissions")
    print("=" * 64)
    print(f"Grant all three to your launching app:  {app}")
    print(f"(TERM_PROGRAM={os.environ.get('TERM_PROGRAM')!r}  bundle={os.environ.get('__CFBundleIdentifier')!r})")
    print("=" * 64)

    results = {
        "Accessibility": check_accessibility(),
        "Screen Recording": check_screen_recording(do_prompt),
        "Microphone": check_microphone(do_prompt),
    }

    all_ok = True
    for name, (status, note) in results.items():
        ok = status == "GRANTED"
        all_ok = all_ok and ok
        mark = "OK " if ok else "!! "
        print(f"  [{mark}] {name:17} {status:14} {note}")

    print("-" * 64)
    if all_ok:
        print(f"All three are granted to {app}. Voice + click should work.")
    else:
        print(f"Fix the ones marked !! by enabling '{app}' in:")
        for name in results:
            if results[name][0] != "GRANTED":
                print(f"    {name:17} System Settings > Privacy & Security > {name}")
        print("After enabling Screen Recording, fully QUIT (Cmd+Q) and relaunch the terminal.")
        print("Re-run with --open to jump straight to the Settings panes.")

    if do_open:
        print("\nopening Settings panes ...")
        for url in PANES.values():
            subprocess.run(["open", url], check=False)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
