"""Tests for the quantization-config resolver (no GPU required)."""

from __future__ import annotations

import pytest

from ais5.quant import QuantConfig, resolve_quant_config


def test_string_presets():
    assert resolve_quant_config("none").kind == "none"
    assert resolve_quant_config("fp16").kind == "none"
    assert resolve_quant_config("bnb8").bits == 8
    assert resolve_quant_config("bnb4").bits == 4


def test_none_returns_default():
    assert resolve_quant_config(None).kind == "none"


def test_passthrough():
    cfg = QuantConfig(name="custom", kind="awq")
    assert resolve_quant_config(cfg) is cfg


def test_dict_with_kind():
    cfg = resolve_quant_config({"kind": "bnb4", "quant_type": "fp4"})
    assert cfg.kind == "bnb4"
    assert cfg.extra.get("quant_type") == "fp4"


def test_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown quant preset"):
        resolve_quant_config("magic-bits")


def test_to_hf_none():
    assert resolve_quant_config("none").to_hf() is None
