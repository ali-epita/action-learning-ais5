#!/usr/bin/env python3
"""Go/no-go preflight for a PointCast live demo.

Checks, in order, and prints one PASS/WARN/FAIL line each:

  1. artifacts   - model + STT dirs resolve from this CWD; optional sha256
                   against the provenance manifest (--hash, ~10s for 2.9 GB)
  2. imports     - mlx, mlx_vlm, PySide6, sounddevice importable in this venv
  3. permissions - Accessibility / Screen Recording / Microphone status
  4. disk        - enough free space for logs/temp
  5. live model  - optional (--ground): load the model and run ONE real
                   grounding call on the current screen, so you walk on stage
                   with Metal kernels already compiled and a measured latency

    .venv-demo/bin/python scripts/pointcast_preflight.py                  # fast checks
    .venv-demo/bin/python scripts/pointcast_preflight.py --hash --ground  # full dress rehearsal

Exit code 0 = go; 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"  [{status}] {name:<22} {detail}")


def check_artifacts(do_hash: bool) -> None:
    from ais5.pointcast.config import PointCastConfig

    cfg = PointCastConfig()
    for label, path in (("grounding model", cfg.model_path), ("STT model", cfg.stt_model)):
        if os.path.isdir(path):
            report(PASS, label, path)
        else:
            report(FAIL, label, f"{path} not found (run from the repo root)")
    if not do_hash:
        return
    import glob
    import hashlib
    from pathlib import Path

    files = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
    if not files:
        report(FAIL, "weights sha256", "no .safetensors in the model dir")
        return
    h = hashlib.sha256()
    with open(files[0], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    digest = h.hexdigest()
    manifest = Path("checkpoints/qwen2.5-vl-3b-lora-r64/_provenance/artifact_sha256.txt")
    if manifest.exists() and digest in manifest.read_text():
        report(PASS, "weights sha256", f"{digest[:16]}… matches provenance")
    elif manifest.exists():
        report(FAIL, "weights sha256", f"{digest[:16]}… NOT in {manifest}")
    else:
        report(WARN, "weights sha256", f"{digest[:16]}… (no provenance manifest to compare)")


def check_imports() -> None:
    for mod in ("mlx", "mlx_vlm", "PySide6", "sounddevice", "mss", "pyautogui", "pynput"):
        try:
            __import__(mod)
            report(PASS, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            report(FAIL, f"import {mod}", repr(e)[:60])


def check_permissions() -> None:
    try:
        import pointcast_permissions as perms
    except Exception as e:  # noqa: BLE001
        report(WARN, "permissions", f"doctor unavailable: {e!r}")
        return
    app = perms.launching_app()
    for name, fn in (
        ("Accessibility", lambda: perms.check_accessibility()),
        ("Screen Recording", lambda: perms.check_screen_recording(False)),
        ("Microphone", lambda: perms.check_microphone(False)),
    ):
        try:
            status, _hint = fn()
            s = str(status).upper()
            if s == "GRANTED" or s.startswith("OK"):
                report(PASS, name, f"granted to {app}")
            elif s == "DENIED":
                report(FAIL, name, f"DENIED — grant to {app}, then quit + relaunch it")
            else:
                report(WARN, name, f"{status} (speak-test with the voice doctor to confirm)")
        except Exception as e:  # noqa: BLE001
            report(WARN, name, f"check failed: {e!r}"[:70])


def check_disk() -> None:
    free_gb = shutil.disk_usage(".").free / 1e9
    report(PASS if free_gb > 5 else WARN, "disk free", f"{free_gb:.1f} GB")


def check_ground() -> None:
    print("  loading the model + one live grounding call (first call compiles Metal, ~60s)...")
    try:
        from ais5.pointcast.backends import get_backend
        from ais5.pointcast.capture import capture_screen

        backend = get_backend("mlx").load()
        frame = capture_screen(1512)
        t0 = time.perf_counter()
        out = backend.predict(frame.image, "the clock in the menu bar")
        dt = time.perf_counter() - t0
        pt = out.parsed.point
        if pt is None:
            report(WARN, "live grounding", f"no parse in {dt:.1f}s: {out.text[:40]!r}")
        else:
            lx, ly = frame.mapper.ground_to_logical(*pt)
            report(PASS, "live grounding", f"({lx:.0f}, {ly:.0f}) logical in {dt:.1f}s — model is warm")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "live grounding", repr(e)[:80])


def main() -> int:
    ap = argparse.ArgumentParser(description="PointCast demo preflight")
    ap.add_argument("--hash", action="store_true", help="sha256 the weights against the provenance manifest")
    ap.add_argument("--ground", action="store_true", help="load the model and run one live grounding call")
    a = ap.parse_args()

    print("PointCast preflight")
    check_artifacts(a.hash)
    check_imports()
    if sys.platform == "darwin":
        check_permissions()
    check_disk()
    if a.ground:
        check_ground()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print(f"\n{'NO-GO' if fails else 'GO'}: {len(fails)} fail, {len(warns)} warn, "
          f"{sum(1 for r in _results if r[0] == PASS)} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
