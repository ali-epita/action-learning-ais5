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
    ap.add_argument("--live", action="store_true",
                    help="enable live voice: adds a ◉ live button to the command bar (and 'start listening' phrases) for a hands-free session until you say 'done'")
    ap.add_argument("--ptt", action="store_true", help="also bind a physical hold-to-talk key (off by default; the mic button is the trigger)")
    ap.add_argument("--ptt-key", default="ctrl_r", help="the hold-to-talk key when --ptt is set (pynput Key name like ctrl_r/f13, or a single char)")
    ap.add_argument("--stt-model", default="parakeet-ctc-110m-asr",
                    help="offline ASR model dir (default: fast Parakeet-CTC-110M)")
    ap.add_argument("--ground-max-side", type=int, default=1512)
    ap.add_argument("--try-twice", action="store_true", help="enable crop-then-click retry")
    ap.add_argument("--retina-crop", action="store_true",
                    help="with --try-twice: crop the native-resolution capture for the re-click stages (more detail on small targets)")
    ap.add_argument("--trust-panel", action="store_true",
                    help="show the trust panel (peak memory + weights sha256 + offline statement)")
    ap.add_argument("--tasks", action="store_true", help="enable multi-step task recipes (off = run single-shot as is)")
    ap.add_argument("--agent", action="store_true",
                    help="enable free-form agent mode: say 'do <goal>' and the model plans each click (max 10 steps, preview + abort per step)")
    ap.add_argument("--deep-links", action="store_true", help="prefer macOS settings deep-links over clicking when a recipe has one")
    ap.add_argument("--dry-run", action="store_true", help="never dispatch a real OS click")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ...")
    ap.add_argument("--log-file", default=None, help="also write logs to this rotating file")
    a = ap.parse_args()

    # Merge saved in-app settings (the overlay's gear panel): an explicitly
    # passed CLI flag wins; otherwise a saved setting overrides the default.
    from .settings import load_settings

    saved = load_settings()

    def merged(cli_value, field, dest):
        if cli_value != ap.get_default(dest):
            return cli_value  # explicitly passed on the command line
        return saved.get(field, cli_value)

    cfg = PointCastConfig(
        backend=a.backend,
        model_path=a.model,
        hotkey=a.hotkey,
        confirm_mode=merged(a.confirm, "confirm_mode", "confirm"),
        countdown_seconds=a.countdown,
        tts_engine=merged(a.tts, "tts_engine", "tts"),
        ground_max_side=a.ground_max_side,
        use_try_twice=merged(a.try_twice, "use_try_twice", "try_twice"),
        retina_crop=merged(a.retina_crop, "retina_crop", "retina_crop"),
        show_trust_panel=merged(a.trust_panel, "show_trust_panel", "trust_panel"),
        enable_tasks=merged(a.tasks, "enable_tasks", "tasks"),
        enable_agent=merged(a.agent, "enable_agent", "agent"),
        use_deep_links=merged(a.deep_links, "use_deep_links", "deep_links"),
        dry_run=merged(a.dry_run, "dry_run", "dry_run"),
        enable_voice=merged(a.voice, "enable_voice", "voice"),
        enable_live_voice=merged(a.live, "enable_live_voice", "live"),
        enable_ptt_key=a.ptt,
        ptt_key=a.ptt_key,
        stt_model=a.stt_model,
        log_level=a.log_level,
        log_file=a.log_file,
    )
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
