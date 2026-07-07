"""Persist the user's in-app settings (the toggles in the overlay's settings
panel) at ``~/.pointcast/settings.json``, next to the recorded recipes.

Precedence at launch: an explicitly passed CLI flag wins over a saved setting,
which wins over the built-in default — so scripted launches stay reproducible
while the panel remembers what you chose interactively.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..utils.logging import get_logger

_log = get_logger("pointcast.settings")

DEFAULT_SETTINGS_PATH = Path.home() / ".pointcast" / "settings.json"

# The settings the panel can change, with their PointCastConfig field names.
# Everything here is safe to toggle at runtime (read at call time, or handled
# explicitly by the controller). Model/hotkey/resolution stay CLI-only.
TOGGLABLE_FIELDS = (
    "enable_voice",
    "enable_live_voice",
    "enable_tasks",
    "enable_agent",
    "use_deep_links",
    "use_try_twice",
    "retina_crop",
    "show_trust_panel",
    "dry_run",
)
CHOICE_FIELDS = {"tts_engine": ("say", "neural", "off"), "confirm_mode": ("countdown", "explicit")}
ALL_FIELDS = TOGGLABLE_FIELDS + tuple(CHOICE_FIELDS)


def _resolve(path: str | Path | None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_SETTINGS_PATH


def load_settings(path: str | Path | None = None) -> dict:
    """Saved settings, or {} — a corrupt/missing file must never block launch."""
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        _log.warning("could not read settings %s: %r", p, e)
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if k in TOGGLABLE_FIELDS and isinstance(v, bool):
            out[k] = v
        elif k in CHOICE_FIELDS and v in CHOICE_FIELDS[k]:
            out[k] = v
    return out


def save_settings(values: dict, path: str | Path | None = None) -> Path:
    """Atomically persist the known settings from ``values``."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in values.items() if k in ALL_FIELDS}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(keep, indent=2))
    tmp.replace(p)
    return p
