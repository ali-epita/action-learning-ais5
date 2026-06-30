"""PointCast runtime configuration.

One dataclass drives the whole app: backend, capture/grounding resolution,
interaction (hotkey, confirm semantics), speech, and visuals. Defaults encode
the decisions in ``AIS5_MVP_Specifications.md`` and the build plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PointCastConfig:
    # ── model / backend ──────────────────────────────────────────────────────
    backend: str = "mlx"  # "mlx" (on-device default) | "gguf" (M6)
    model_path: str = "mlx-qwen-r64-4bit"
    max_tokens: int = 64

    # ── capture / grounding resolution ───────────────────────────────────────
    # Downscale the captured screen so its long side <= this before grounding.
    # Retina screens are huge (e.g. 3024px); Qwen2.5-VL slows down and loses
    # accuracy on oversized images. Coordinates are mapped back to the real
    # screen afterwards. Set to 0/None to disable.
    ground_max_side: int = 1512
    monitor_index: int = 1  # mss monitor: 1 = primary (0 = whole virtual desktop)

    # ── interaction ──────────────────────────────────────────────────────────
    hotkey: str = "<alt>+<space>"  # pynput GlobalHotKeys syntax
    confirm_mode: str = "countdown"  # "countdown" (auto-click unless cancelled) | "explicit"
    countdown_seconds: float = 3.0

    # ── Try-Twice (M2; spine runs a single call unless enabled) ──────────────
    use_try_twice: bool = False
    crop_sizes: tuple[int, ...] = (768, 512)
    gate_displacement_frac: float = 0.06  # coarse↔refined agreement threshold, as a fraction of the crop size (tune via calibration)

    # ── disambiguation (M3) ──────────────────────────────────────────────────
    disambig_samples: int = 4  # stochastic samples drawn when the ladder yields <2 distinct candidates
    disambig_temperature: float = 0.7
    disambig_cluster_tol: float = 40.0  # grounding-px radius for merging candidate points
    max_candidates: int = 4

    # ── speech ───────────────────────────────────────────────────────────────
    tts_engine: str = "say"  # "say" (macOS native) | "neural" (mlx_audio, M4) | "off"
    tts_voice: str | None = None
    speak_locator: bool = True

    # ── visuals ──────────────────────────────────────────────────────────────
    crosshair_color: str = "#16a34a"
    crosshair_radius: int = 28
    crosshair_thickness: int = 4
    hud_bg: str = "#111827"
    hud_fg: str = "#f9fafb"
    marker_color: str = "#f59e0b"  # disambiguation markers (M3)

    # ── voice input (M4: STT) ───────────────────────────────────────────────
    enable_voice: bool = False  # enable voice input (mic button in the search bar)
    enable_ptt_key: bool = False  # also bind a physical hold-to-talk key (off by default; mic button is the trigger)
    ptt_key: str = "ctrl_r"  # the hold-to-talk key when enable_ptt_key is set (pynput Key name, or a single char)
    stt_model: str = "whisper-large-v3-turbo-asr-fp16"  # local offline dir (repo root); resolved as a local path first
    stt_samplerate: int = 16000
    stt_min_seconds: float = 0.3  # ignore recordings shorter than this
    tts_neural_model: str = "prince-canuma/Kokoro-82M"  # neural TTS (showcase); falls back to `say`
    tts_neural_voice: str = "af_heart"

    # ── logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"  # set "DEBUG" for more detail (or env AIS5_LOG_LEVEL)
    log_file: str | None = None  # also write logs to this rotating file if set

    # ── click actuation ──────────────────────────────────────────────────────
    # macOS only delivers a click to a background window's control once that
    # window is key (first-mouse rule). Bring the window under the target to the
    # front before clicking so the first click actuates instead of just focusing.
    activate_target_window: bool = True
    activate_settle_ms: int = 180  # wait after activating, before the click lands

    # ── task mode (multi-step recipes) ───────────────────────────────────────
    # Off by default: the app runs "as is" (single-shot grounding, one click per
    # request). With it on, a request that matches a recipe (e.g. "how much free
    # storage") runs a short multi-step sequence and speaks the answer, while
    # everything else still single-clicks — so both modes are available at once.
    enable_tasks: bool = False
    use_deep_links: bool = False  # prefer a recipe's macOS settings deep-link over clicking, when present
    task_preview_ms: int = 800  # show each step's crosshair this long before clicking (lets the user abort)

    # ── safety ───────────────────────────────────────────────────────────────
    dry_run: bool = False  # if True, never dispatch a real OS click (log instead)

    @property
    def ground_max_side_or_none(self) -> int | None:
        return self.ground_max_side or None
