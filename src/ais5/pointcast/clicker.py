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

    def right_click(self, x: float, y: float) -> None:
        if self.dry_run:
            print(f"[pointcast dry-run] right click at logical ({x:.0f}, {y:.0f})")
            return
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.moveTo(x, y, duration=self.move_duration)
        pyautogui.click(button="right")

    def move(self, x: float, y: float) -> None:
        if self.dry_run:
            return
        import pyautogui

        pyautogui.moveTo(x, y, duration=self.move_duration)

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        """Type ``text`` into whatever field currently has focus (the caller
        clicks the field first). No-op in dry-run."""
        if self.dry_run:
            print(f"[pointcast dry-run] type {text!r}")
            return
        import pyautogui

        pyautogui.write(text, interval=interval)

    def press_enter(self) -> None:
        if self.dry_run:
            print("[pointcast dry-run] press enter")
            return
        import pyautogui

        pyautogui.press("enter")

    def press_keys(self, keys: tuple[str, ...]) -> None:
        """Press a single key or a chord (modifiers first, main key last),
        e.g. ("enter",) or ("command", "space"). No-op on an empty tuple."""
        if not keys:
            return
        if self.dry_run:
            print(f"[pointcast dry-run] press {'+'.join(keys)}")
            return
        import pyautogui

        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
