"""Map the user's words to a recipe (Stage 1: simple, specific substring match).

Kept deliberately conservative so it does not hijack ordinary single-click
targets: only multi-word trigger phrases match, and the most specific (longest)
matching phrase wins. Swap in embeddings or a small classifier later without
changing callers.
"""

from __future__ import annotations

from .recipes import RECIPES, Recipe


def match_recipe(text: str, recipes: tuple[Recipe, ...] = RECIPES) -> Recipe | None:
    t = (text or "").lower().strip()
    if not t:
        return None
    best: tuple[Recipe, int] | None = None
    for r in recipes:
        for phrase in r.utterances:
            if phrase in t and (best is None or len(phrase) > best[1]):
                best = (r, len(phrase))
    return best[0] if best else None
