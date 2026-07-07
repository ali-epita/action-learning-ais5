"""configure_caches redirects cache env vars without clobbering user choices."""

from __future__ import annotations

import os

from ais5.utils.env import configure_caches, free_model

_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME")


def test_sets_unset_vars(tmp_path, monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    root = tmp_path / "cache"

    resolved = configure_caches(str(root))

    assert os.environ["HF_HOME"] == str(root)
    assert os.path.isdir(os.environ["HF_HUB_CACHE"])
    assert resolved["HF_DATASETS_CACHE"].startswith(str(root))


def test_respects_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    for v in _VARS[1:]:
        monkeypatch.delenv(v, raising=False)

    resolved = configure_caches(str(tmp_path / "cache"))

    assert os.environ["HF_HOME"] == "/custom/hf"
    assert resolved["HF_HOME"] == "/custom/hf"


def test_free_model_without_cuda_is_safe():
    free_model(object())  # no torch CUDA on CI; must not raise
