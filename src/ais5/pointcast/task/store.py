"""Persist user-recorded recipes as JSON and merge them with the built-ins.

A recorded recipe is plain data: a name plus the ordered target phrases the user
demonstrated. On replay each phrase is re-grounded by the model, so a recording
stays valid even when the UI shifts a little. Stored at ``cfg.recipes_path``
(default ``~/.pointcast/recipes.json``), so it survives restarts and is offline.
"""

from __future__ import annotations

import json
import os
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
        "steps": [
            {"target": s.target, "say": s.say, "settle_ms": s.settle_ms, "expect": s.expect}
            for s in r.steps
        ],
        "deep_link": r.deep_link,
        "question": r.question,
    }


def recipe_from_dict(d: dict) -> Recipe:
    """Build a Recipe from stored JSON, validating as it goes. Raises ValueError
    on a malformed entry so the loader can skip just that entry: a hand-edited
    null utterance or numeric target must not crash recipe matching later (it
    runs on every submitted phrase) or poison the whole store."""
    name = d.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"recipe name must be a non-empty string, got {name!r}")
    name = name.strip()
    utterances = tuple(
        u.strip() for u in (d.get("utterances") or ()) if isinstance(u, str) and u.strip()
    ) or (name.lower(),)
    steps = []
    for s in d.get("steps", ()):
        target = s.get("target") if isinstance(s, dict) else None
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"step target must be a non-empty string, got {s!r}")
        say = s.get("say", "")
        expect = s.get("expect", "")
        steps.append(Step(
            target=target.strip(),
            say=say if isinstance(say, str) else "",
            settle_ms=int(s.get("settle_ms", 900)),
            expect=expect if isinstance(expect, str) else "",
        ))
    deep_link = d.get("deep_link")
    question = d.get("question", "")
    return Recipe(
        name=name,
        utterances=utterances,
        steps=tuple(steps),
        deep_link=deep_link if isinstance(deep_link, str) else None,
        question=question if isinstance(question, str) else "",
    )


def load_user_recipes(path: str | Path | None = None) -> list[Recipe]:
    p = _resolve(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 — a corrupt file must never crash the app
        _log.warning("could not read recipes %s: %r", p, e)
        return []
    out: list[Recipe] = []
    for d in data if isinstance(data, list) else []:
        try:
            out.append(recipe_from_dict(d))
        except Exception as e:  # noqa: BLE001 — skip the bad entry, keep the rest
            _log.warning("skipping malformed recipe entry in %s: %r", p, e)
    return out


def save_user_recipe(recipe: Recipe, path: str | Path | None = None) -> Path:
    """Write ``recipe`` to the store, replacing any existing recipe of the same
    name (case-insensitive, matching how recipes are looked up)."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in load_user_recipes(path) if r.name.lower() != recipe.name.lower()]
    payload = json.dumps([recipe_to_dict(r) for r in (*kept, recipe)], indent=2)
    # Atomic write (tmp + rename): a crash mid-write must not corrupt the whole
    # store — load_user_recipes would then silently return [] forever. The tmp
    # name is per-process so two instances saving at once cannot trip over one
    # shared temp file.
    tmp = p.with_suffix(f"{p.suffix}.{os.getpid()}.tmp")
    tmp.write_text(payload)
    tmp.replace(p)
    _log.info("saved recipe %r (%d steps) -> %s", recipe.name, len(recipe.steps), p)
    return p


def all_recipes(path: str | Path | None = None) -> tuple[Recipe, ...]:
    """Built-in recipes plus user recipes; a user recipe overrides a built-in of
    the same name (case-insensitive — matching is case-insensitive too, so a
    recipe saved as "Check Storage" must shadow the built-in "check storage")."""
    user = load_user_recipes(path)
    user_names = {r.name.lower() for r in user}
    return (*(r for r in RECIPES if r.name.lower() not in user_names), *user)
