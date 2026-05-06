"""Click instruction templates.

We default to the same template across every model to keep the comparison fair
(per the proposal's "same prompt/action format" constraint). Models that need
their own dialect (e.g. PaliGemma's `<loc####>` tokens) override at call time.
"""

from __future__ import annotations

CLICK_PROMPT = (
    "You are a GUI agent. Look at the screenshot and decide where to click "
    "to complete the user's instruction.\n"
    "Reply with EXACTLY one click in the format: <click>x, y</click>\n"
    "Coordinates are pixels measured from the top-left of the image.\n\n"
    "Instruction: {instruction}"
)


def format_click_prompt(instruction: str, *, template: str = CLICK_PROMPT) -> str:
    return template.format(instruction=instruction)
