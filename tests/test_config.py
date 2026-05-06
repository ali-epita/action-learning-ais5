"""Tests for YAML config loading + env-var expansion."""

from __future__ import annotations

from ais5.utils.config import load_config


def test_loads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb: [x, y]\n")
    assert load_config(p) == {"a": 1, "b": ["x", "y"]}


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("AIS5_TEST_VAR", "hello")
    p = tmp_path / "c.yaml"
    p.write_text("greeting: ${AIS5_TEST_VAR}\nfallback: ${AIS5_NOT_SET:-default}\n")
    cfg = load_config(p)
    assert cfg["greeting"] == "hello"
    assert cfg["fallback"] == "default"
    monkeypatch.delenv("AIS5_TEST_VAR", raising=False)


def test_empty_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p) == {}
