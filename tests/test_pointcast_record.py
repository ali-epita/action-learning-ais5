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


def test_parse_save_as_command():
    assert parse_command("save recipe as open bluetooth") == ("save_as", "open bluetooth")
    assert parse_command("save as open bluetooth") == ("save_as", "open bluetooth")
    assert parse_command("save this as quick check") == ("save_as", "quick check")
    assert parse_command("save task as morning routine") == ("save_as", "morning routine")
    # the plain forms are untouched
    assert parse_command("save recipe") == ("save", None)
    assert parse_command("save the recipe") == ("save", None)


def test_parse_save_with_readback():
    assert parse_command("save recipe and read the battery percentage") == \
        ("save", "the battery percentage")
    assert parse_command("save recipe and say the free storage") == ("save", "the free storage")
    assert parse_command("done recording and tell me the wifi network") == \
        ("save", "the wifi network")
    # save-as with a read-back packs both into "name :: readback"
    assert parse_command("save recipe as show battery and read the battery percentage") == \
        ("save_as", "show battery :: the battery percentage")
    # no read-back -> unchanged
    assert parse_command("save recipe as show battery") == ("save_as", "show battery")


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


def test_store_skips_malformed_entries_keeps_good_ones():
    import json

    path = os.path.join(tempfile.mkdtemp(), "recipes.json")
    good = recipe_to_dict(Recipe(name="open mail", utterances=("open mail",),
                                 steps=(Step("Mail in the Dock"),)))
    null_utterance = {"name": "bad one", "utterances": [None], "steps": [{"target": "x"}]}
    numeric_target = {"name": "bad two", "utterances": ["bad two"], "steps": [{"target": 42}]}
    no_name = {"utterances": ["z"], "steps": []}
    with open(path, "w") as f:
        json.dump([good, null_utterance, numeric_target, no_name], f)

    loaded = load_user_recipes(path)
    # the null utterance is recoverable (falls back to the name); the numeric
    # target and the missing name are not - those entries are skipped, the rest load
    assert [r.name for r in loaded] == ["open mail", "bad one"]
    assert loaded[1].utterances == ("bad one",)
    # and matching over the merged set must not crash on what was loaded
    assert match_recipe("open mail", all_recipes(path)).name == "open mail"


def test_store_override_and_replace_are_case_insensitive():
    path = os.path.join(tempfile.mkdtemp(), "recipes.json")
    # a user recipe saved as "Check Storage" must shadow the built-in
    # "check storage" (matching is case-insensitive, so override must be too)
    save_user_recipe(Recipe(name="Check Storage", utterances=("check storage",),
                            steps=(Step("General"),)), path)
    merged = all_recipes(path)
    mine = [r for r in merged if r.name.lower() == "check storage"]
    assert len(mine) == 1 and mine[0].steps[0].target == "General"
    assert match_recipe("check storage", merged).steps[0].target == "General"
    # re-saving under a different case replaces instead of duplicating
    save_user_recipe(Recipe(name="CHECK STORAGE", utterances=("check storage",),
                            steps=(Step("Other"),)), path)
    assert len(load_user_recipes(path)) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} record tests passed")
