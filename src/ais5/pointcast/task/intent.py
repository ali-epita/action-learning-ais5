"""Map the user's words to a recipe (Stage 1: conservative phrase matching).

Deliberately conservative so it does not hijack ordinary single-click targets:

  - only MULTI-word trigger phrases are considered (single-word utterances like
    "wi-fi" would match inside "the wi-fi settings icon");
  - phrases match on word boundaries, not raw substrings ("storage space" does
    not match "storage spaces");
  - anything starting with "click " is treated as a literal click target and
    never matched against recipes (the escape hatch: "click check storage"
    grounds the words instead of running the recipe);
  - the most specific (longest) matching phrase wins.

Swap in embeddings or a small classifier later without changing callers.
"""

from __future__ import annotations

import re

from .recipes import RECIPES, Recipe


def _phrase_matches(phrase: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def match_recipe(text: str, recipes: tuple[Recipe, ...] = RECIPES) -> Recipe | None:
    t = (text or "").lower().strip()
    if not t:
        return None
    if t.startswith("click "):
        return None  # explicit literal target — never a recipe
    best: tuple[Recipe, int] | None = None
    for r in recipes:
        for phrase in r.utterances:
            if not isinstance(phrase, str):
                continue  # defensive: a malformed store entry must not crash matching
            p = phrase.lower().strip()
            if len(p.split()) < 2:
                continue  # single-word triggers are too hijack-prone
            if _phrase_matches(p, t) and (best is None or len(p) > best[1]):
                best = (r, len(p))
    return best[0] if best else None
