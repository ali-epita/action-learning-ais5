"""Tests for the click/read/type verb command parser (pure, no Qt)."""

from __future__ import annotations

from ais5.pointcast.command import CommandIntent, detect_verb, parse_command, read_prompt


# ── click (default + explicit) ────────────────────────────────────────────────
def test_plain_text_is_a_click_target():
    c = parse_command("the search box")
    assert c.kind == "click" and c.target == "the search box" and not c.explicit_click


def test_explicit_click_is_literal():
    c = parse_command("click the Send button")
    assert c.kind == "click" and c.target == "the Send button" and c.explicit_click
    assert parse_command("click on the gear icon").target == "the gear icon"


# ── read ──────────────────────────────────────────────────────────────────────
def test_read_lead():
    c = parse_command("read my final kaggle grade")
    assert c.kind == "read" and c.query == "my final kaggle grade" and c.verb == "read"


def test_read_question_forms():
    assert parse_command("what is my total score").kind == "read"
    assert parse_command("what's the battery percentage").kind == "read"
    assert parse_command("how much free storage is there").kind == "read"


def test_read_prompt_builds_a_question():
    p = read_prompt("my final kaggle grade")
    assert "my final kaggle grade" in p and p.rstrip().endswith("value.")
    assert read_prompt("what is my score").count("?") == 1


# ── type ──────────────────────────────────────────────────────────────────────
def test_type_with_target():
    c = parse_command("type hello world into the search box")
    assert c.kind == "type" and c.text == "hello world" and c.target == "the search box"
    assert not c.submit


def test_type_in_fallback():
    c = parse_command("type ali.cherri in the username field")
    assert c.kind == "type" and c.text == "ali.cherri" and c.target == "the username field"


def test_type_without_target_uses_focused():
    c = parse_command("type hello there")
    assert c.kind == "type" and c.text == "hello there" and c.target == ""


def test_type_with_submit():
    for phrase in (
        "type mlx into the search box and press enter",
        "type mlx into the search box and submit",
        "type mlx into the search box then hit return",
    ):
        c = parse_command(phrase)
        assert c.kind == "type" and c.text == "mlx" and c.target == "the search box" and c.submit


def test_type_strips_surrounding_quotes():
    c = parse_command('type "hello world" into the box')
    assert c.text == "hello world"


# ── live verb detection (for the input chip) ──────────────────────────────────
def test_detect_verb_on_complete_word():
    assert detect_verb("read ") == "read"
    assert detect_verb("read my grade") == "read"
    assert detect_verb("type foo") == "type"
    assert detect_verb("click the box") == "click"
    assert detect_verb("what is my score") == "read"


def test_detect_verb_none_until_word_complete():
    assert detect_verb("rea") is None
    assert detect_verb("reading glasses") is None  # not the verb
    assert detect_verb("the search box") is None
    assert detect_verb("") is None


def test_bare_verbs_carry_nothing_to_act_on():
    # A clipped dictation like just "click" must not become a click target
    # named "click" - it comes back empty so the controller reprompts.
    for spoken, kind in (("click", "click"), ("read", "read"), ("type", "type"),
                         ("Click", "click"), ("click.", "click")):
        c = parse_command(spoken)
        assert c.kind == kind
        assert (c.target, c.query, c.text) == ("", "", "")


def test_press_single_keys():
    for spoken, key in (("press enter", "enter"), ("press return", "enter"),
                        ("hit escape", "esc"), ("press the space bar", "space"),
                        ("press delete", "backspace"), ("press tab", "tab"),
                        ("press page down", "pagedown"), ("press up arrow", "up")):
        c = parse_command(spoken)
        assert c.kind == "press" and c.keys == (key,), (spoken, c.keys)


def test_press_chords_in_spoken_and_typed_forms():
    assert parse_command("press command and space").keys == ("command", "space")
    assert parse_command("press cmd+shift+4").keys == ("command", "shift", "4")
    assert parse_command("press control and c").keys == ("ctrl", "c")
    assert parse_command("press option and up arrow").keys == ("option", "up")


def test_press_unknown_key_is_refused_not_guessed():
    c = parse_command("press the any key")
    assert c.kind == "press" and c.keys == ()
    assert parse_command("press flurb").keys == ()
    # bare "press" reprompts like the other bare verbs
    bare = parse_command("press")
    assert bare.kind == "press" and bare.keys == () and bare.text == ""


def test_right_click_variants():
    for spoken in ("right click the file icon", "right-click the file icon",
                   "right click on the file icon"):
        c = parse_command(spoken)
        assert c.kind == "click" and c.button == "right", spoken
        assert c.target == "the file icon"
        assert c.explicit_click  # never a recipe
    # plain click stays a left click
    assert parse_command("click the file icon").button == "left"
    # bare "right click" has nothing to ground -> empty target reprompts
    bare = parse_command("right click")
    assert bare.kind == "click" and bare.button == "right" and bare.target == ""


def test_press_does_not_shadow_type_submit_tail():
    c = parse_command("type hello into the search box and press enter")
    assert c.kind == "type" and c.text == "hello" and c.submit


def test_detect_verb_press_and_right_click():
    assert detect_verb("press ") == "press"
    assert detect_verb("right click ") == "click"
    assert detect_verb("pressing") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} command tests passed")
