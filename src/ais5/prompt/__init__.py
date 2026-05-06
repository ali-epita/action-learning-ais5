"""Prompt formatting and unified action-output parsing."""

from .action import ParsedAction, parse_click
from .templates import CLICK_PROMPT, format_click_prompt

__all__ = [
    "CLICK_PROMPT",
    "ParsedAction",
    "format_click_prompt",
    "parse_click",
]
