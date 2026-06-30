"""Parse the spoken/typed commands that drive record-by-demonstration.

  "record check storage"        -> ("record", "check storage")
  "remember this as open mail"  -> ("record", "open mail")
  "save recipe" / "done recording" -> ("save", None)
  "cancel recording"            -> ("cancel", None)

Anything else returns None and is treated as an ordinary target.
"""

from __future__ import annotations

import re

_RECORD = re.compile(
    r"^(?:record(?:\s+recipe)?|start recording|remember this as|remember as)\s+(.+)$", re.IGNORECASE
)
_SAVE = re.compile(r"^(?:save(?:\s+the)?\s+recipe|done recording|stop recording|finish recording)$", re.IGNORECASE)
_CANCEL = re.compile(r"^(?:cancel recording|discard recording|stop recording without saving)$", re.IGNORECASE)


def parse_command(text: str) -> tuple[str, str | None] | None:
    t = (text or "").strip().rstrip(".")
    m = _RECORD.match(t)
    if m:
        return ("record", m.group(1).strip())
    if _SAVE.match(t):
        return ("save", None)
    if _CANCEL.match(t):
        return ("cancel", None)
    return None
