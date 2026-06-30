"""OS-level click dispatch via pyautogui (logical screen coordinates).

Requires macOS Accessibility permission. The controller hides the crosshair
overlay immediately before calling ``click`` so the dispatch lands on the real
target, not on our window.
"""

from __future__ import annotations


class Clicker:
    def __init__(self, dry_run: bool = False, move_duration: float = 0.12):
        self.dry_run = dry_run
        self.move_duration = move_duration

    def click(self, x: float, y: float) -> None:
        if self.dry_run:
            print(f"[pointcast dry-run] click at logical ({x:.0f}, {y:.0f})")
            return
        import pyautogui

        pyautogui.FAILSAFE = True  # slam cursor to a corner to abort
        pyautogui.moveTo(x, y, duration=self.move_duration)
        pyautogui.click()

    def move(self, x: float, y: float) -> None:
        if self.dry_run:
            return
        import pyautogui

        pyautogui.moveTo(x, y, duration=self.move_duration)
