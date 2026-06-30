"""Persist user-recorded recipes as JSON and merge them with the built-ins.

A recorded recipe is plain data: a name plus the ordered target phrases the user
demonstrated. On replay each phrase is re-grounded by the model, so a recording
stays valid even when the UI shifts a little. Stored at ``cfg.recipes_path``
(default ``~/.pointcast/recipes.json``), so it survives restarts and is offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...utils.logging import get_logger
from .recipes import RECIPES, Recipe, Step

_log = get_logger("pointcast.task.store")

DEFAULT_RECIPES_PATH = Path.home() / ".pointcast" / "recipes.json"


def _resolve(path: str | Path | None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_RECIPES_PATH


def recipe_to_dict(r: Recipe) -> dict:
    return {
        "name": r.name,
        "utterances": list(r.utterances),
        "steps": [{"target": s.target, "say": s.say, "settle_ms": s.settle_ms} for s in r.steps],
        "deep_link": r.deep_link,
        "question": r.question,
    }


def recipe_from_dict(d: dict) -> Recipe:
    return Recipe(
        name=d["name"],
        utterances=tuple(d.get("utterances") or (d["name"].lower(),)),
        steps=tuple(
            Step(target=s["target"], say=s.get("say", ""), settle_ms=int(s.get("settle_ms", 900)))
            for s in d.get("steps", [])
        ),
        deep_link=d.get("deep_link"),
        question=d.get("question", ""),
    )


def load_user_recipes(path: str | Path | None = None) -> list[Recipe]:
    p = _resolve(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return [recipe_from_dict(d) for d in data]
    except Exception as e:  # noqa: BLE001 — a corrupt file must never crash the app
        _log.warning("could not read recipes %s: %r", p, e)
        return []


def save_user_recipe(recipe: Recipe, path: str | Path | None = None) -> Path:
    """Write ``recipe`` to the store, replacing any existing recipe of the same name."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in load_user_recipes(path) if r.name != recipe.name]
    p.write_text(json.dumps([recipe_to_dict(r) for r in (*kept, recipe)], indent=2))
    _log.info("saved recipe %r (%d steps) -> %s", recipe.name, len(recipe.steps), p)
    return p


def all_recipes(path: str | Path | None = None) -> tuple[Recipe, ...]:
    """Built-in recipes plus user recipes; a user recipe overrides a built-in of
    the same name."""
    user = load_user_recipes(path)
    user_names = {r.name for r in user}
    return (*(r for r in RECIPES if r.name not in user_names), *user)
