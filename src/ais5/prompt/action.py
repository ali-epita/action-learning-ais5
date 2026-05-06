"""Best-effort parser that extracts a click point from any model's response.

Each model speaks its own dialect:
    Qwen2.5-VL    "<click>123, 456</click>"  or JSON-like
    PaliGemma     "<loc0123><loc0456>"      (1024-bin, row then col)
    OS-Atlas      "(123, 456)"               or JSON {"x": .., "y": ..}
    ShowUI        "{action: 'click', x: 123, y: 456}"
    Box outputs   "<box>x1,y1,x2,y2</box>"   → centroid

`parse_click` tries them in order. Returns `point=None` if nothing parseable
was found. Coordinates returned are in image pixels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Order matters: most specific patterns first to avoid false positives.
_QWEN_CLICK = re.compile(
    r"<click>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</click>",
    re.IGNORECASE,
)
_BOX_TAG = re.compile(
    r"<box>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*</box>",
    re.IGNORECASE,
)
_LOC_TOKENS = re.compile(r"<loc(\d{4})><loc(\d{4})>")
_JSON_XY = re.compile(
    r'"x"\s*:\s*(-?\d+(?:\.\d+)?)[\s\S]{0,40}?"y"\s*:\s*(-?\d+(?:\.\d+)?)'
    r"|"
    r'"y"\s*:\s*(-?\d+(?:\.\d+)?)[\s\S]{0,40}?"x"\s*:\s*(-?\d+(?:\.\d+)?)',
)
_PIXEL_TUPLE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


@dataclass(frozen=True)
class ParsedAction:
    point: tuple[float, float] | None
    bbox: tuple[float, float, float, float] | None = None
    raw: str = ""
    parser: str = "none"


def parse_click(
    text: str,
    image_size: tuple[int, int] | None = None,
    *,
    paligemma_bins: int = 1024,
) -> ParsedAction:
    """Extract a click point from `text`. Returns `point=None` if not found."""
    if not isinstance(text, str):
        text = str(text)

    if m := _QWEN_CLICK.search(text):
        return ParsedAction(
            point=(float(m.group(1)), float(m.group(2))),
            raw=text,
            parser="qwen-click",
        )

    if m := _BOX_TAG.search(text):
        x1, y1, x2, y2 = (float(g) for g in m.groups())
        return ParsedAction(
            point=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            bbox=(x1, y1, x2, y2),
            raw=text,
            parser="box-tag",
        )

    if (m := _LOC_TOKENS.search(text)) and image_size is not None:
        # PaliGemma emits <loc{row}><loc{col}> indices into a `paligemma_bins`-bin grid.
        row_bin, col_bin = int(m.group(1)), int(m.group(2))
        w, h = image_size
        y = (row_bin + 0.5) / paligemma_bins * h
        x = (col_bin + 0.5) / paligemma_bins * w
        return ParsedAction(point=(x, y), raw=text, parser="paligemma-loc")

    # Try JSON object first (handles {"x": .., "y": ..} regardless of key order).
    if (json_obj := _try_extract_json(text)) is not None:
        x = json_obj.get("x")
        y = json_obj.get("y")
        if x is None or y is None:
            # Some models nest under "action": {"x": .., "y": ..}.
            inner = json_obj.get("action")
            if isinstance(inner, dict):
                x, y = inner.get("x"), inner.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return ParsedAction(point=(float(x), float(y)), raw=text, parser="json-object")

    if m := _JSON_XY.search(text):
        if m.group(1) is not None:
            x, y = float(m.group(1)), float(m.group(2))
        else:
            y, x = float(m.group(3)), float(m.group(4))
        return ParsedAction(point=(x, y), raw=text, parser="json-xy")

    if m := _PIXEL_TUPLE.search(text):
        return ParsedAction(
            point=(float(m.group(1)), float(m.group(2))),
            raw=text,
            parser="pixel-tuple",
        )

    return ParsedAction(point=None, raw=text, parser="none")


def _try_extract_json(text: str) -> dict | None:
    """Find the first `{...}` block and try to JSON-decode it."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        result = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(result, dict):
                        return result
                    break
        start = text.find("{", start + 1)
    return None
