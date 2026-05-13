"""Tests for ScreenSpot loader compatibility with HF mirror split naming."""

from __future__ import annotations

import sys
from types import ModuleType

from PIL import Image

from ais5.data.screenspot import _load_iter, _row_to_sample


def test_screenspot_falls_back_to_train_split(monkeypatch):
    calls: list[str] = []

    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(repo_id, *, split, streaming=False, **kwargs):
        calls.append(split)
        if split == "test":
            raise ValueError('Unknown split "test". Should be one of [\'train\'].')
        return [
            {
                "image": Image.new("RGB", (100, 100)),
                "bbox": [10, 10, 20, 20],
                "instruction": "submit",
            }
        ]

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    samples = list(_load_iter("mirror/repo", "screenspot-v2", "test"))

    assert calls == ["test", "train"]
    assert len(samples) == 1
    assert samples[0].extra["requested_split"] == "test"
    assert samples[0].extra["loaded_split"] == "train"


def test_screenspot_v2_xywh_bbox_is_converted_to_xyxy():
    sample = _row_to_sample(
        {
            "image": Image.new("RGB", (1000, 800)),
            "img_filename": "example.png",
            "bbox": [223, 78, 601, 593],
            "instruction": "check the weather",
        },
        "screenspot-v2",
        0,
        bbox_format="xywh",
    )

    assert sample.bbox == (223, 78, 824, 671)
    assert sample.sample_id == "example.png"
