"""Grid-resume helpers: read JSONL results and skip completed cells."""

from __future__ import annotations

import json

from ais5.quant.recipes import resolve_quant_config
from ais5.utils.io import completed_keys, read_jsonl


def test_read_jsonl_missing(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_completed_keys(tmp_path):
    p = tmp_path / "grid.jsonl"
    rows = [
        {"model": "qwen2.5-vl-3b", "quant": "fp16", "benchmark": "screenspot-v2"},
        {"model": "showui-2b", "quant": "bnb-8bit", "benchmark": "screenspot-pro"},
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    done = completed_keys(p, ("model", "quant", "benchmark"))
    assert ("qwen2.5-vl-3b", "fp16", "screenspot-v2") in done
    assert ("showui-2b", "bnb-8bit", "screenspot-pro") in done
    assert ("showui-2b", "bnb-4bit-nf4", "screenspot-pro") not in done


def test_resume_key_uses_quant_name():
    # The JSONL stores qc.name, not the spec string. Resume must key on it.
    assert resolve_quant_config("none").name == "fp16"
    assert resolve_quant_config("bnb8").name == "bnb-8bit"
    assert resolve_quant_config("bnb4").name == "bnb-4bit-nf4"
