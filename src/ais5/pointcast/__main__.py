"""``python -m ais5.pointcast`` — launch the on-device pointer.

    .venv-demo/bin/python -m ais5.pointcast                  # MLX, real clicks
    .venv-demo/bin/python -m ais5.pointcast --dry-run        # no real clicks (safe test)
    .venv-demo/bin/python -m ais5.pointcast --try-twice      # enable crop-then-click retry
    .venv-demo/bin/python -m ais5.pointcast --confirm explicit --tts off
"""

from __future__ import annotations

import argparse

from .app import run
from .config import PointCastConfig


def main() -> int:
    ap = argparse.ArgumentParser(prog="ais5.pointcast", description="PointCast — on-device accessibility pointer")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "gguf"])
    ap.add_argument("--model", default="mlx-qwen-r64-4bit")
    ap.add_argument("--hotkey", default="<alt>+<space>", help="pynput GlobalHotKeys syntax")
    ap.add_argument("--confirm", default="countdown", choices=["countdown", "explicit"])
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--tts", default="say", choices=["say", "neural", "off"])
    ap.add_argument("--voice", action="store_true", help="enable voice input (mic button in the search bar)")
    ap.add_argument("--ptt", action="store_true", help="also bind a physical hold-to-talk key (off by default; the mic button is the trigger)")
    ap.add_argument("--ptt-key", default="ctrl_r", help="the hold-to-talk key when --ptt is set (pynput Key name like ctrl_r/f13, or a single char)")
    ap.add_argument("--stt-model", default="whisper-large-v3-turbo-asr-fp16")
    ap.add_argument("--ground-max-side", type=int, default=1512)
    ap.add_argument("--try-twice", action="store_true", help="enable crop-then-click retry")
    ap.add_argument("--dry-run", action="store_true", help="never dispatch a real OS click")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ...")
    ap.add_argument("--log-file", default=None, help="also write logs to this rotating file")
    a = ap.parse_args()

    cfg = PointCastConfig(
        backend=a.backend,
        model_path=a.model,
        hotkey=a.hotkey,
        confirm_mode=a.confirm,
        countdown_seconds=a.countdown,
        tts_engine=a.tts,
        ground_max_side=a.ground_max_side,
        use_try_twice=a.try_twice,
        dry_run=a.dry_run,
        enable_voice=a.voice,
        enable_ptt_key=a.ptt,
        ptt_key=a.ptt_key,
        stt_model=a.stt_model,
        log_level=a.log_level,
        log_file=a.log_file,
    )
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
