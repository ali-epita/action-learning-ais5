# PointCast — on-device accessibility pointer (MVP)

Say or type where to click; PointCast grounds it to a pixel on your **live
screen** with the LoRA-r64 Qwen2.5-VL model (85.5% on ScreenSpot-V2), shows a
high-contrast crosshair, **speaks** a confirmation with a countdown, and
dispatches a **real OS click** — fully offline. Built on the `ais5` library;
see `AIS5_MVP_Specifications.md` and `PROVENANCE.md`.

## Status

| Milestone | State |
|---|---|
| M0 Foundations (adapter, provenance, MLX backend) | done |
| M1 Core pointer spine (hotkey -> capture -> ground -> crosshair -> spoken confirm -> real click) | done, validated on-device |
| M2 Full Try-Twice (768->512 ladder + stability gate + badge) | done |
| M3 Disambiguation (numbered candidates, pick by number) | done |
| M4 Voice I/O (push-to-talk STT + neural TTS, both opt-in) | done |
| M5 Trust panel | next |
| M6 GGUF backend | next |
| M7 calibration + polish | next |

> Threading note: all MLX work (warmup, grounding, transcription) runs on a
> single dedicated inference thread. MLX GPU streams are thread-bound, so calling
> MLX from multiple threads concurrently crashes ("no Stream(gpu) in current
> thread" / segfault). This is why warmup and grounding are serialized.

## One-time setup (macOS permissions)

PointCast needs two permissions. Grant them to the program you launch it from
(Terminal / iTerm, or your IDE). **System Settings → Privacy & Security →**

1. **Screen Recording** — to capture the screen for grounding.
2. **Accessibility** — to move/click the mouse and to use the global hotkey.

You'll likely be prompted on first run; after granting, fully quit and relaunch
the terminal app.

Dependencies are already installed in `.venv-demo` (PySide6, pyautogui, mss,
pynput, mlx, mlx_vlm; plus mlx_audio and sounddevice for voice input).

## Run

Always start with **`--dry-run`** (no real clicks — it shows the crosshair and
speaks the locator, but never clicks, so you can check accuracy safely):

```bash
cd "/Users/ali/Desktop/Action Learning/action-learning-ais5"
.venv-demo/bin/python -m ais5.pointcast --dry-run
```

Then the real thing:

```bash
.venv-demo/bin/python -m ais5.pointcast
```

Flow: press the hotkey (**⌥-Space** by default) → a search bar appears → type the
target (e.g. "the search bar"), or speak it (see Voice input below) → Enter → the
box hides, the screen is captured, the model grounds it → a green crosshair
appears, the locator is spoken, a countdown starts → it clicks unless you cancel.

During the countdown: **Enter** = click now · **Esc** = cancel · **R** = retry.

## Voice input (speak instead of typing) — offline

Enable voice with `--voice`. Voice lives **inside the search bar**, triggered by
its **mic button**:

1. Press **⌥-Space** to open the search bar (it shows a **mic button** on the right).
2. **Click the mic button** and speak the target (e.g. "the search bar"). While
   listening, the mic pulses red and the bar is outlined in red.
3. **Click the mic again** to stop — the speech is transcribed on-device and
   **typed into the search field** for you.
4. Review it and press **Enter** to point, exactly as if you had typed it.

```bash
.venv-demo/bin/python -m ais5.pointcast --dry-run --voice    # ⌥-Space, then click the mic
```

Speech-to-text uses a **local, fully offline** MLX Whisper model shipped in the
repo root at `whisper-large-v3-turbo-asr-fp16/` (no network, no HF download). It
is resolved as a local path first, so run from the repo dir (or pass an absolute
path / a different model via `--stt-model`). Transcription is ~3 s and shares the
single MLX inference thread with grounding.

Voice input needs **Microphone** permission in addition to Screen Recording and
Accessibility (System Settings → Privacy & Security). If the mic or model is
unavailable, voice fails gracefully and you can always type.

**Optional physical key:** add `--ptt` to also bind a hold-to-talk key (default
Right Control; change with `--ptt-key`). It is off by default — the mic button is
the trigger.

## Task mode (multi-step, optional)

By default the app runs "as is": one request, one click. Add `--tasks` and the
app also handles short multi-step goals, while ordinary single-click requests
still work, so both modes are available at once.

```bash
.venv-demo/bin/python -m ais5.pointcast --tasks --voice    # single-click AND multi-step
```

Say or type a goal like "how much free storage do I have". A matching **recipe**
runs a short, fixed sequence (Settings, then General, then Storage), grounding
each step with the same on-device model, then reads the answer off the final
screen and **speaks it** ("you have 120 GB available"). The procedure is a
curated recipe, not a runtime planner, which keeps it reliable and fits the 3B
model on 8 GB. Press the hotkey during a task to cancel it; in `--dry-run` it
narrates and shows each crosshair without clicking.

Built-in recipes (`src/ais5/pointcast/task/recipes.py`): check storage, check
battery, check wifi. Where macOS exposes a settings deep-link, `--deep-links`
jumps straight to the pane instead of clicking through.

### Record your own (record-by-demonstration)

Instead of hand-writing a recipe, demonstrate it once and PointCast remembers it:

1. Say or type **"record &lt;name&gt;"** (e.g. "record open mail"). PointCast starts
   recording.
2. Do the steps normally (hotkey, name each target, it clicks). Each successful
   click is captured as a step; a target it cannot find is not recorded.
3. Say or type **"save recipe"** (or "done recording"). It is saved and is
   immediately usable by name.

Recordings are stored as JSON at `~/.pointcast/recipes.json` (set with
`recipes_path`) and merged with the built-ins at startup; a recording reuses your
target phrases and re-grounds them each run, so it survives small UI changes.
This is the recommended way to add multi-step tasks on 8 GB: no planner model,
and no guessing at target phrasings.

## Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--dry-run` | off | never dispatch a real click (safe testing) |
| `--confirm explicit` | `countdown` | require Enter to click (never auto-click) |
| `--countdown 3.0` | 3.0s | auto-click delay |
| `--tts say\|neural\|off` | `say` | `say` = macOS voice (instant); `neural` = mlx_audio Kokoro (showcase, falls back to `say`); `off` = silent |
| `--voice` | off | enable voice input (mic button in the search bar; needs Microphone permission) |
| `--ptt` | off | also bind a physical hold-to-talk key (the mic button is the default trigger) |
| `--ptt-key ctrl_r` | `ctrl_r` | which hold-to-talk key when `--ptt` is set |
| `--ground-max-side 1512` | 1512 | downscale the screen before grounding (lower = faster, less precise) |
| `--try-twice` | off | enable the full Try-Twice retry ladder (~2 calls, ~18s) |
| `--tasks` | off | enable multi-step task recipes (off = single-shot as is) |
| `--deep-links` | off | prefer a recipe's macOS settings deep-link over clicking, when present |
| `--hotkey "<cmd>+<shift>+p"` | `<alt>+<space>` | pynput GlobalHotKeys syntax |

## Notes / known characteristics

- **First interaction is slow** (~60s): MLX compiles Metal kernels once. The app
  warms up in the background at startup, so wait for "ready" before the first use.
  After warmup, expect ~9s per grounding call on this machine (hardware-dependent).
- Multi-monitor: M1 targets the **primary** display.
- **Background windows**: macOS only actuates a click on a window once it is key
  (the first-mouse rule), so PointCast first brings the window under the target
  to the front, then clicks — the target app comes into focus as it is clicked.
  Disable with `activate_target_window=False`, or tune `activate_settle_ms` (the
  pause after raising, before the click) if a slow app needs longer to come up.
- Quantization is a **privacy/offline** win, not a speed win (memory-bound at batch 1).

## Self-tests (no display / permissions / model needed)

```bash
.venv-demo/bin/python tests/test_pointcast_geometry.py     # coordinate math + locator
.venv-demo/bin/python scripts/pointcast_ui_selftest.py     # full controller wiring (offscreen)
.venv-demo/bin/python scripts/pointcast_smoke.py           # real MLX predict + crop_then_click (loads the model)
```
