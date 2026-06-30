"""Headless tests for window_focus.window_owner_at (pure dict logic, no screen)."""

from __future__ import annotations

from ais5.pointcast import window_focus as wf


def _wins():
    return [
        # high-layer menu bar spanning the whole top — must be ignored
        {"kCGWindowLayer": 25, "kCGWindowOwnerPID": 1, "kCGWindowOwnerName": "MenuBar",
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 2000, "Height": 24}},
        # front normal window
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 2, "kCGWindowOwnerName": "Front",
         "kCGWindowBounds": {"X": 100, "Y": 100, "Width": 400, "Height": 300}},
        # larger window behind it
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 3, "kCGWindowOwnerName": "Back",
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 800, "Height": 600}},
    ]


def test_picks_topmost_normal_window():
    wf._windows_front_to_back = _wins
    assert wf.window_owner_at(150, 150) == (2, "Front")  # inside Front (front-to-back order)


def test_falls_through_to_lower_window():
    wf._windows_front_to_back = _wins
    assert wf.window_owner_at(700, 550) == (3, "Back")  # only Back covers it


def test_ignores_high_layer_chrome():
    wf._windows_front_to_back = _wins
    # (10,10) is under the menu bar AND Back; the layer-0 Back wins
    assert wf.window_owner_at(10, 10) == (3, "Back")


def test_skip_pid_is_skipped():
    wf._windows_front_to_back = _wins
    assert wf.window_owner_at(150, 150, skip_pid=2) == (3, "Back")


def test_none_when_outside_all_windows():
    wf._windows_front_to_back = _wins
    assert wf.window_owner_at(1900, 1900) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} window-focus tests passed")
