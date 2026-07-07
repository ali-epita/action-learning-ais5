"""Parse a spoken/typed request into a command with a leading verb.

PointCast understands these verbs, each a first-class action:

    click <target>                     -> click the described element (default)
    right click <target>               -> secondary click on the element
    read <thing>  /  what is <thing>   -> read a value off the screen, speak it
    type <text> [into <target>]        -> click the field and type; optional submit
         [and press enter | and submit]
    press <key(s)>                     -> press a key or chord on the keyboard
                                          ("press enter", "press command and space")

Anything with no recognised verb is treated as a click target (the historical
behaviour), so plain "the search box" still clicks. An explicit "click ..."
prefix marks a literal click target (it never matches a task recipe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# verbs recognised for the glowing chip + routing
VERBS = ("click", "read", "type", "press")

_READ_LEAD = re.compile(r"^\s*read\s+(?:me\s+|out\s+|the\s+)?(.+)$", re.IGNORECASE)
_READ_QUESTION = re.compile(r"^\s*(what(?:'s| is| are| does)?|how (?:much|many)|tell me)\b.*", re.IGNORECASE)
_TYPE_LEAD = re.compile(r"^\s*type\s+(.+)$", re.IGNORECASE | re.DOTALL)
_CLICK_LEAD = re.compile(r"^\s*click\s+(?:on\s+)?(.+)$", re.IGNORECASE | re.DOTALL)
_RCLICK_LEAD = re.compile(r"^\s*right[\s-]?click\s+(?:on\s+)?(.+)$", re.IGNORECASE | re.DOTALL)
_PRESS_LEAD = re.compile(r"^\s*(?:press|hit)\s+(.+)$", re.IGNORECASE)
# trailing "... and press enter / and submit / then hit return"
_SUBMIT_TAIL = re.compile(
    r"\s*(?:,?\s*(?:and|then)\s+(?:press|hit|tap)\s+(?:enter|return)|,?\s*and\s+submit|\s+then\s+submit)\s*$",
    re.IGNORECASE,
)
# split "type X into Y" / "type X in Y" — prefer "into", then " in "
_INTO = re.compile(r"^(.*?)\s+into\s+(.+)$", re.IGNORECASE | re.DOTALL)
_IN = re.compile(r"^(.*?)\s+in\s+(.+)$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CommandIntent:
    kind: str  # "click" | "read" | "type" | "press"
    target: str = ""  # click/type: element to click / field to type into ("" = focused)
    query: str = ""  # read: the thing to read off the screen
    text: str = ""  # type: the literal text to type (original case preserved)
    submit: bool = False  # type: press Enter after typing
    explicit_click: bool = False  # user wrote "click ..." (literal, never a recipe)
    verb: str = "click"  # the detected leading verb (for the UI chip)
    button: str = "left"  # click: "left" | "right" (secondary click)
    keys: tuple[str, ...] = ()  # press: normalized pyautogui key names, chord order


# Spoken/typed key names -> pyautogui key names. Modifiers and the keys that
# matter for driving macOS by voice; anything unrecognized is refused (the
# controller reprompts) rather than guessed — a wrong key press can submit a
# form or close a window.
_KEY_ALIASES = {
    "enter": "enter", "return": "enter",
    "escape": "esc", "esc": "esc",
    "space": "space", "spacebar": "space",
    "tab": "tab",
    "delete": "backspace", "backspace": "backspace",
    "command": "command", "cmd": "command",
    "control": "ctrl", "ctrl": "ctrl",
    "option": "option", "alt": "option",
    "shift": "shift",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end",
    "pageup": "pageup", "pagedown": "pagedown",
}
# Connector/filler words dictation adds freely: "command AND space",
# "the escape KEY", "up ARROW", "space BAR".
_KEY_FILLERS = {"and", "then", "the", "a", "key", "keys", "arrow", "bar"}
_FN_KEY = re.compile(r"^f([1-9]|1[0-2])$")


def normalize_keys(body: str) -> tuple[str, ...]:
    """Parse a spoken key chord ("command and space", "cmd+shift+4") into
    pyautogui key names, in the order spoken. Returns () when any part is not
    a known key — the caller must reprompt, never guess."""
    tokens = [t for t in re.split(r"[-+,/]|\s+", body.lower()) if t and t not in _KEY_FILLERS]
    # join the two-word keys dictation produces ("page down" -> pagedown)
    joined: list[str] = []
    for t in tokens:
        if joined and joined[-1] == "page" and t in ("up", "down"):
            joined[-1] = f"page{t}"
        else:
            joined.append(t)
    out: list[str] = []
    for t in joined:
        if t in _KEY_ALIASES:
            out.append(_KEY_ALIASES[t])
        elif (len(t) == 1 and t.isalnum()) or _FN_KEY.match(t):
            out.append(t)
        else:
            return ()
    return tuple(out)


def detect_verb(text: str) -> str | None:
    """The leading verb for the live input chip, or None if no verb typed yet.
    Matches whole leading words so 'read'/'type'/'click' light up as soon as the
    word is complete (followed by a space) — 'reading' does not."""
    t = (text or "").lstrip().lower()
    for prefix in ("right click", "right-click"):
        if t == prefix or t.startswith(prefix + " "):
            return "click"
    for v in VERBS:
        if t == v or t.startswith(v + " "):
            return v
    if _READ_QUESTION.match(t):
        return "read"
    return None


def parse_command(text: str) -> CommandIntent:
    raw = (text or "").strip()
    if not raw:
        return CommandIntent(kind="click", target="")

    # A bare verb (a clipped dictation like just "click") carries nothing to act
    # on: return it with an empty target/query/text so the controller reprompts,
    # instead of treating the verb itself as a click target.
    bare = raw.lower().rstrip(".!,?")
    if bare in VERBS:
        return CommandIntent(kind=bare, verb=bare)
    if bare in ("right click", "right-click"):
        return CommandIntent(kind="click", button="right", verb="click")

    if m := _TYPE_LEAD.match(raw):
        body = m.group(1).strip()
        submit = False
        if s := _SUBMIT_TAIL.search(body):
            submit = True
            body = body[: s.start()].strip()
        target = ""
        typed = body
        if mm := _INTO.match(body):
            typed, target = mm.group(1).strip(), mm.group(2).strip()
        elif mm := _IN.match(body):
            typed, target = mm.group(1).strip(), mm.group(2).strip()
        typed = typed.strip().strip('"').strip("'")
        return CommandIntent(kind="type", text=typed, target=target, submit=submit, verb="type")

    # "press <keys>" — checked before the read/click fallbacks so "press enter"
    # is a key press, not a click target. Keys the map does not know come back
    # as keys=() with the raw phrase in .text, so the controller can say so.
    if m := _PRESS_LEAD.match(raw):
        body = m.group(1).strip()
        return CommandIntent(kind="press", keys=normalize_keys(body), text=body, verb="press")

    if m := _READ_LEAD.match(raw):
        return CommandIntent(kind="read", query=m.group(1).strip(), verb="read")
    if _READ_QUESTION.match(raw):
        return CommandIntent(kind="read", query=raw, verb="read")

    # right click before click: "right click ..." must not fall through to a
    # left click on a target literally named "right click the file".
    if m := _RCLICK_LEAD.match(raw):
        return CommandIntent(kind="click", target=m.group(1).strip(), explicit_click=True,
                             button="right", verb="click")
    if m := _CLICK_LEAD.match(raw):
        return CommandIntent(kind="click", target=m.group(1).strip(), explicit_click=True, verb="click")

    return CommandIntent(kind="click", target=raw, verb="click")


def read_prompt(query: str) -> str:
    """Turn a read request into a VQA prompt answered off the current screen."""
    q = query.strip().rstrip("?")
    if re.match(r"^(what|how|is|are|does|do)\b", q, re.IGNORECASE):
        question = q + "?"
    else:
        question = f"What is {q}?"
    return (
        f"Look at this screenshot and answer from what is visible. {question} "
        "Answer in one short, natural sentence with the value."
    )
