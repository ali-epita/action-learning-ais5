"""Headless tests for record-by-demonstration: command parsing + recipe store."""

from __future__ import annotations

import os
import tempfile

from ais5.pointcast.task import (
    Recipe, Step, all_recipes, load_user_recipes, match_recipe, parse_command, save_user_recipe,
)
from ais5.pointcast.task.store import recipe_from_dict, recipe_to_dict


def test_parse_command():
    assert parse_command("record check storage") == ("record", "check storage")
    assert parse_command("remember this as open mail") == ("record", "open mail")
    assert parse_command("save recipe") == ("save", None)
    assert parse_command("done recording") == ("save", None)
    assert parse_command("cancel recording") == ("cancel", None)
    assert parse_command("the search box") is None
    assert parse_command("") is None


def test_recipe_dict_roundtrip():
    r = Recipe(name="x", utterances=("x", "y"),
               steps=(Step("a", say="A", settle_ms=500),), deep_link="dl", question="q?")
    assert recipe_from_dict(recipe_to_dict(r)) == r


def test_store_save_load_merge_and_replace():
    path = os.path.join(tempfile.mkdtemp(), "recipes.json")
    save_user_recipe(Recipe(name="open mail", utterances=("open mail",), steps=(Step("Mail in the Dock"),)), path)

    loaded = load_user_recipes(path)
    assert len(loaded) == 1 and loaded[0].name == "open mail"

    names = {r.name for r in all_recipes(path)}
    assert {"open mail", "check storage", "check battery", "check wifi"} <= names  # built-ins + user

    # saving the same name replaces, not appends
    save_user_recipe(Recipe(name="open mail", utterances=("open mail",),
                            steps=(Step("Mail in the Dock"), Step("Inbox"))), path)
    again = load_user_recipes(path)
    assert len(again) == 1 and len(again[0].steps) == 2


def test_user_recipe_overrides_builtin_and_is_matchable():
    path = os.path.join(tempfile.mkdtemp(), "recipes.json")
    save_user_recipe(Recipe(name="check storage", utterances=("my storage",), steps=(Step("X"),)), path)
    merged = all_recipes(path)
    storage = [r for r in merged if r.name == "check storage"]
    assert len(storage) == 1 and storage[0].utterances == ("my storage",)  # overridden
    assert match_recipe("show my storage please", merged).name == "check storage"


def test_missing_store_is_empty_not_an_error():
    assert load_user_recipes(os.path.join(tempfile.mkdtemp(), "nope.json")) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} record tests passed")
