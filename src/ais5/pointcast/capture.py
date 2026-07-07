"""Live screen capture via mss, with the coordinate mapper for click-back.

Captures the primary monitor in physical pixels, optionally downscales for
grounding, and bundles a ``CoordinateMapper`` so predicted points map back to
real logical screen coordinates. Requires macOS Screen-Recording permission.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .geometry import CoordinateMapper, compute_ground_scale


@dataclass
class CaptureFrame:
    image: Image.Image  # grounding image (RGB, possibly downscaled) — fed to the model
    full_image: Image.Image  # full physical-resolution capture (RGB)
    mapper: CoordinateMapper


def _logical_size(fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        import pyautogui

        w, h = pyautogui.size()
        return (int(w), int(h))
    except Exception:
        return fallback


def capture_screen(ground_max_side: int | None = 1512, monitor_index: int = 1) -> CaptureFrame:
    """Grab the screen and return a CaptureFrame ready for grounding + click-back.

    KNOWN LIMIT (M1 scope): the click-back mapping is only validated for the
    PRIMARY display. For other monitors, ``pyautogui.size()`` still reports the
    primary display's logical size and mss origins mix coordinate spaces, so
    points can land offset. A warning is logged when a non-primary monitor is
    requested.
    """
    import mss

    with mss.mss() as sct:
        monitors = sct.monitors
        idx = monitor_index if 0 <= monitor_index < len(monitors) else 1
        if idx != 1:
            from ..utils.logging import get_logger

            get_logger("pointcast.capture").warning(
                "monitor_index=%d: click mapping is only validated for the primary "
                "display; expect offsets on secondary monitors", idx,
            )
        mon = monitors[idx]
        raw = sct.grab(mon)
        full = Image.frombytes("RGB", raw.size, raw.rgb)

    cap_w, cap_h = full.size
    log_w, log_h = _logical_size(fallback=(cap_w, cap_h))
    f = compute_ground_scale(cap_w, cap_h, ground_max_side)
    if f < 1.0:
        gimg = full.resize((max(1, round(cap_w * f)), max(1, round(cap_h * f))), Image.LANCZOS)
    else:
        gimg = full

    mapper = CoordinateMapper(
        capture_size=(cap_w, cap_h),
        logical_size=(log_w, log_h),
        ground_scale=f,
        offset_physical=(int(mon.get("left", 0)), int(mon.get("top", 0))),
    )
    return CaptureFrame(image=gimg, full_image=full, mapper=mapper)
