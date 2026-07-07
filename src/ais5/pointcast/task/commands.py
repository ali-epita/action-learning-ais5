"""Parse the spoken/typed commands that drive record-by-demonstration and
saving a finished agent run as a recipe.

  "record check storage"        -> ("record", "check storage")
  "remember this as open mail"  -> ("record", "open mail")
  "save recipe" / "done recording" -> ("save", None)
  "save recipe and read the battery percentage"
                                -> ("save", "the battery percentage")   # read-back
  "save recipe as open bluetooth" / "save as open bluetooth"
                                -> ("save_as", "open bluetooth")
  "save recipe as show battery and read the battery percentage"
                                -> ("save_as", "show battery :: the battery percentage")
  "cancel recording"            -> ("cancel", None)

Anything else returns None and is treated as an ordinary target.
"""

from __future__ import annotations

import re

_RECORD = re.compile(
    r"^(?:record(?:\s+recipe)?|start recording|remember this as|remember as)\s+(.+)$", re.IGNORECASE
)
# Optional trailing read-back: "... and read/answer/tell me/say <question>".
_READBACK = r"(?:\s+(?:and\s+)?(?:read|answer|tell me|say|speak)\s+(.+))?"
_SAVE_AS = re.compile(
    r"^save\s+(?:this\s+|that\s+)?(?:recipe\s+|task\s+)?as\s+(.+?)" + _READBACK + r"$", re.IGNORECASE
)
_SAVE = re.compile(
    r"^(?:save(?:\s+the)?\s+recipe|done recording|stop recording|finish recording)"
    + _READBACK + r"$", re.IGNORECASE
)
_CANCEL = re.compile(r"^(?:cancel recording|discard recording|stop recording without saving)$", re.IGNORECASE)


def parse_command(text: str) -> tuple[str, str | None] | None:
    t = (text or "").strip().rstrip(".")
    m = _RECORD.match(t)
    if m:
        return ("record", m.group(1).strip())
    m = _SAVE_AS.match(t)
    if m:
        name = m.group(1).strip()
        readback = (m.group(2) or "").strip()
        return ("save_as", f"{name} :: {readback}" if readback else name)
    m = _SAVE.match(t)
    if m:
        readback = (m.group(1) or "").strip()
        return ("save", readback or None)
    if _CANCEL.match(t):
        return ("cancel", None)
    return None
