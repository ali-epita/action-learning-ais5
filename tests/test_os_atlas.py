"""OS-Atlas loader: pairs annotation JSON with the image zip and denormalizes."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

from PIL import Image

from ais5.adapt import adapt_os_atlas_row, os_atlas


def _png_bytes(w, h):
    buf = BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _build(tmp_path, entries, *, img_name="screenshots/a.png", size=(100, 200)):
    zpath = tmp_path / "imgs.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(img_name, _png_bytes(*size))
    jpath = tmp_path / "ann.json"
    jpath.write_text(json.dumps(entries))
    return zpath, jpath


def _patch(monkeypatch, zpath, jpath):
    def fake_dl(repo_id, filename, **kw):
        return str(jpath) if filename.endswith(".json") else str(zpath)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    monkeypatch.setitem(os_atlas._SUBSETS, "t", ("ann.json", ["imgs.zip"]))


def test_nested_elements_denormalized(tmp_path, monkeypatch):
    entries = [{
        "img_filename": "screenshots/a.png",
        "elements": [
            {"instruction": "click x", "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"instruction": "click y", "bbox": [0.5, 0.5, 0.7, 0.6]},
        ],
    }]
    zpath, jpath = _build(tmp_path, entries)
    _patch(monkeypatch, zpath, jpath)

    rows = list(os_atlas.stream_os_atlas(["t"]))
    assert len(rows) == 2
    # 100x200 image: bbox [0.1,0.2,0.3,0.4] -> [10,40,30,80] pixels
    assert rows[0]["bbox"] == [10.0, 40.0, 30.0, 80.0]
    ex = adapt_os_atlas_row(rows[0])
    assert ex is not None
    assert ex.target_point == (20.0, 60.0)  # bbox center


def test_flat_entry_and_limit(tmp_path, monkeypatch):
    entries = [
        {"img_filename": "screenshots/a.png", "instruction": "go", "bbox": [0.0, 0.0, 1.0, 1.0]},
        {"img_filename": "screenshots/a.png", "instruction": "go2", "bbox": [0.1, 0.1, 0.2, 0.2]},
    ]
    zpath, jpath = _build(tmp_path, entries)
    _patch(monkeypatch, zpath, jpath)

    rows = list(os_atlas.stream_os_atlas("t", limit=1))
    assert len(rows) == 1
    assert rows[0]["instruction"] == "go"


def test_unknown_subset_raises():
    import pytest

    with pytest.raises(ValueError):
        list(os_atlas.stream_os_atlas(["does-not-exist"]))
