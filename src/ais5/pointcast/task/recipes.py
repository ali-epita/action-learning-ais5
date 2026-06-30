"""Curated multi-step recipes: the procedure is encoded, not planned at runtime.

Each recipe is a short ordered list of grounding instructions (one per click),
an optional macOS deep-link shortcut, and an optional final question that is
read off the resulting screen and spoken back. This is the 8 GB-friendly design:
no planner model, just the existing 3B grounder plus its free-form ``ask``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    target: str  # grounding instruction for this click (fed to the grounder)
    say: str = ""  # short spoken narration before the click ("Opening General")
    settle_ms: int = 900  # wait after the click for the UI to update before the next step


@dataclass(frozen=True)
class Recipe:
    name: str
    utterances: tuple[str, ...]  # lowercased trigger phrases matched against the user's words
    steps: tuple[Step, ...] = ()
    deep_link: str | None = None  # macOS x-apple.systempreferences: URL (used only if use_deep_links)
    question: str = ""  # final VQA, read off the screen and spoken (empty = navigate only)


CHECK_STORAGE = Recipe(
    name="check storage",
    utterances=(
        "how much free storage", "how much storage", "free storage",
        "storage space", "disk space", "free space", "free disk", "storage left",
    ),
    steps=(
        Step(target="the System Settings icon in the Dock", say="Opening Settings", settle_ms=1500),
        Step(target="General in the settings sidebar", say="Opening General", settle_ms=1000),
        Step(target="Storage in the General list", say="Opening Storage", settle_ms=1200),
    ),
    # Best-effort shortcut; only used when use_deep_links is on (verify per macOS version).
    deep_link="x-apple.systempreferences:com.apple.settings.Storage",
    question=(
        "This is the macOS Storage settings screen. How much storage is free or available? "
        "Answer in one short sentence with the number."
    ),
)

RECIPES: tuple[Recipe, ...] = (CHECK_STORAGE,)
