"""Tests for the persisted in-app settings store (pure, no Qt)."""

from __future__ import annotations

import json
import os
import tempfile

from ais5.pointcast.settings import load_settings, save_settings


def _path():
    return os.path.join(tempfile.mkdtemp(), "settings.json")


def test_roundtrip():
    p = _path()
    save_settings({"enable_voice": True, "use_deep_links": True, "tts_engine": "off"}, p)
    assert load_settings(p) == {"enable_voice": True, "use_deep_links": True, "tts_engine": "off"}


def test_unknown_and_invalid_values_are_dropped():
    p = _path()
    save_settings({"enable_voice": True, "hotkey": "<cmd>+p", "model_path": "evil"}, p)
    assert load_settings(p) == {"enable_voice": True}  # only known fields persist
    # invalid types/choices in a hand-edited file are ignored
    with open(p, "w") as f:
        json.dump({"enable_voice": "yes", "tts_engine": "loud", "dry_run": True}, f)
    assert load_settings(p) == {"dry_run": True}


def test_missing_or_corrupt_file_is_empty():
    assert load_settings(os.path.join(tempfile.mkdtemp(), "nope.json")) == {}
    p = _path()
    with open(p, "w") as f:
        f.write("{not json")
    assert load_settings(p) == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} settings tests passed")
