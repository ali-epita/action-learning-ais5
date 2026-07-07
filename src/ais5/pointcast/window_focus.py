"""Bring the window under the click point to the front before clicking (macOS).

The problem: on macOS a synthetic click on a window that is not the key window
only *activates* it — it does not actuate the control under the cursor. This is
the first-mouse rule (``-[NSView acceptsFirstMouse:]`` defaults to NO), and most
controls do not opt in. PointCast always clicks a background window (its own
confirm HUD was frontmost a moment earlier), so a single click would just raise
the target and the button would not press.

The fix: find the app that owns the topmost normal window at the target point,
raise that specific window (Accessibility ``AXRaise``) and activate its app, wait
a short settle, then dispatch the real click — which now lands on the key window
and actuates. Everything here is best-effort and macOS-only; it returns False (so
the caller just clicks normally) off macOS or on any error.
"""

from __future__ import annotations

import sys

from ..utils.logging import get_logger

_log = get_logger("pointcast.focus")


def _windows_front_to_back() -> list[dict]:
    """On-screen windows, front-to-back, excluding desktop elements."""
    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return list(Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or [])


def window_owner_at(x: float, y: float, skip_pid: int | None = None):
    """(pid, name) of the topmost *normal* (layer 0) window containing (x, y).

    Layer 0 excludes the menu bar, Dock, and floating overlays; ``skip_pid`` lets
    the caller ignore its own windows. Returns None if nothing normal is hit.
    """
    for w in _windows_front_to_back():
        if w.get("kCGWindowLayer", 0) != 0:
            continue
        pid = w.get("kCGWindowOwnerPID")
        if skip_pid is not None and pid == skip_pid:
            continue
        b = w.get("kCGWindowBounds") or {}
        bx, by = b.get("X", 0.0), b.get("Y", 0.0)
        bw, bh = b.get("Width", 0.0), b.get("Height", 0.0)
        if bx <= x < bx + bw and by <= y < by + bh:
            return pid, w.get("kCGWindowOwnerName")
    return None


def _raise_ax_window_at(x: float, y: float) -> None:
    """Best-effort: raise the specific window under (x, y) so the click lands on
    the right one when the owning app has several windows."""
    try:
        import HIServices

        sysw = HIServices.AXUIElementCreateSystemWide()
        err, el = HIServices.AXUIElementCopyElementAtPosition(sysw, float(x), float(y), None)
        if err != 0 or el is None:
            return
        err, win = HIServices.AXUIElementCopyAttributeValue(el, "AXWindow", None)
        if err == 0 and win is not None:
            HIServices.AXUIElementPerformAction(win, "AXRaise")
    except Exception as e:  # noqa: BLE001
        _log.debug("AX raise skipped: %r", e)


def focus_window_at(x: float, y: float, skip_pid: int | None = None) -> bool:
    """Activate the app/window under (x, y) so a following click actuates it.

    Returns True if a *different* app was activated (the caller should then wait
    ``activate_settle_ms`` before clicking); False if the target was already
    frontmost, nothing was under the point, or we are not on macOS.
    """
    if sys.platform != "darwin":
        return False
    try:
        import AppKit

        hit = window_owner_at(x, y, skip_pid)
        if hit is None:
            return False
        pid, name = hit
        ws = AppKit.NSWorkspace.sharedWorkspace()
        front = ws.frontmostApplication()
        if front is not None and front.processIdentifier() == pid:
            return False  # already key — a normal click actuates
        app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            return False
        _raise_ax_window_at(x, y)
        opts = getattr(AppKit, "NSApplicationActivateIgnoringOtherApps", 1 << 1)
        try:
            app.activateWithOptions_(opts)
        except Exception:  # noqa: BLE001 — deprecated on macOS 14+, fall back
            try:
                app.activate()
            except Exception:  # noqa: BLE001
                return False
        _log.info("activated %r (pid %s) under the target before clicking", name, pid)
        return True
    except Exception as e:  # noqa: BLE001
        _log.debug("focus_window_at skipped: %r", e)
        return False
