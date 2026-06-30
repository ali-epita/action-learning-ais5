"""Plain-language locator for a screen point — the non-visual confirmation.

Turns a click point into speech like "the Send button, bottom right" so a
low-vision user can confirm *before* the click lands. A coarse 3×3 region grid
plus an approximate "% down" reading; refined with crop metadata when available.
"""

from __future__ import annotations


def _third(v: float, span: float) -> int:
    if span <= 0:
        return 1
    r = v / span
    return 0 if r < 1 / 3 else (2 if r > 2 / 3 else 1)


def describe_region(x: float, y: float, w: int, h: int) -> str:
    """e.g. 'top left', 'center', 'bottom right'."""
    horiz = ["left", "center", "right"][_third(x, w)]
    vert = ["top", "middle", "bottom"][_third(y, h)]
    if horiz == "center" and vert == "middle":
        return "center"
    if vert == "middle":
        return f"center {horiz}" if horiz != "center" else "center"
    if horiz == "center":
        return f"{vert} center"
    return f"{vert} {horiz}"


def confirmation_phrase(target: str, x: float, y: float, w: int, h: int) -> str:
    """The sentence spoken/shown before clicking."""
    region = describe_region(x, y, w, h)
    target = (target or "").strip()
    if target:
        return f"Clicking {target} — {region}."
    return f"Clicking {region}."


def countdown_hint(seconds: float, *, confirm_mode: str) -> str:
    if confirm_mode == "explicit":
        return "Press Enter to click · Esc cancel · R retry"
    return f"Clicking in {seconds:.0f}s — Esc cancel · R retry · Enter now"
