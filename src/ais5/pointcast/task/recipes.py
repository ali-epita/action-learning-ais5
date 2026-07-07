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
    # Optional pre-step screen check: a short thing that must be VISIBLE before
    # this step grounds (e.g. "the System Settings window"). If the model says
    # it is not visible, the runner waits and rechecks once, then stops safely
    # instead of clicking on the wrong screen (the cascade-failure killer).
    expect: str = ""


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
        "check storage", "check my storage", "my storage",
        "how much free storage", "how much storage", "free storage",
        "storage space", "disk space", "free space", "free disk", "storage left",
    ),
    steps=(
        Step(target="the gray gear wheel icon in the Dock", say="Opening Settings", settle_ms=4000),
        Step(target="General in the settings sidebar", say="Opening General", settle_ms=1000,
             expect="the System Settings window with its sidebar"),
        Step(target="Storage in the General list", say="Opening Storage", settle_ms=1200),
    ),
    # Best-effort shortcut; only used when use_deep_links is on (verify per macOS version).
    deep_link="x-apple.systempreferences:com.apple.settings.Storage",
    question=(
        "This is the macOS Storage settings screen. How much storage is free or available? "
        "Answer in one short sentence with the number."
    ),
)

CHECK_BATTERY = Recipe(
    name="check battery",
    utterances=(
        "battery percentage", "battery level", "battery health", "battery status",
        "how much battery", "check battery", "my battery",
    ),
    steps=(
        Step(target="the gray gear wheel icon in the Dock", say="Opening Settings", settle_ms=4000),
        Step(target="Battery in the settings sidebar", say="Opening Battery", settle_ms=1200,
             expect="the System Settings window with its sidebar"),
    ),
    deep_link="x-apple.systempreferences:com.apple.settings.Battery",
    question=(
        "This is the macOS Battery settings screen. What is the current battery charge "
        "percentage or battery health? Answer in one short sentence."
    ),
)

CHECK_WIFI = Recipe(
    name="check wifi",
    utterances=(
        "which wifi", "what wifi", "wifi network", "wifi status", "check wifi",
        "am i connected", "which network", "wi-fi",
    ),
    steps=(
        Step(target="the gray gear wheel icon in the Dock", say="Opening Settings", settle_ms=4000),
        Step(target="Wi-Fi in the settings sidebar", say="Opening Wi-Fi", settle_ms=1200,
             expect="the System Settings window with its sidebar"),
    ),
    deep_link="x-apple.systempreferences:com.apple.wifi-settings-extension",
    question=(
        "This is the macOS Wi-Fi settings screen. Is Wi-Fi on, and which network is connected? "
        "Answer in one short sentence."
    ),
)

RECIPES: tuple[Recipe, ...] = (CHECK_STORAGE, CHECK_BATTERY, CHECK_WIFI)
