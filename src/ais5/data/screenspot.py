"""ScreenSpot-V2 and ScreenSpot-Pro loaders.

Both benchmarks ship on the Hugging Face Hub. Schemas are slightly different
between the two (and between mirrors), so `_row_to_sample` is defensive and
uses fallback keys.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .types import GroundingSample

# Default mirrors. Override via the env vars or by passing `repo_id` directly.
SCREENSPOT_V2_REPO = "OS-Copilot/ScreenSpot-v2"
SCREENSPOT_PRO_REPO = "likaixin/ScreenSpot-Pro"
_SCREENSPOT_V2_JSONS = (
    "screenspot_desktop_v2.json",
    "screenspot_mobile_v2.json",
    "screenspot_web_v2.json",
)


def _coerce_bbox(raw: Any, *, fmt: str | None = None) -> tuple[float, float, float, float]:
    """Accept (x1, y1, x2, y2) or (x, y, w, h) lists/tuples; return (x1, y1, x2, y2)."""
    if raw is None:
        raise ValueError("missing bbox")
    coords = [float(v) for v in raw]
    if len(coords) != 4:
        raise ValueError(f"expected 4 bbox values, got {len(coords)}: {raw}")
    x1, y1, a, b = coords
    if fmt == "xywh":
        return (x1, y1, x1 + a, y1 + b)
    if fmt == "xyxy":
        return (x1, y1, a, b)
    if a < x1 or b < y1:
        return (x1, y1, x1 + a, y1 + b)
    return (x1, y1, a, b)


def _first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _row_to_sample(
    row: dict[str, Any],
    benchmark: str,
    idx: int,
    *,
    bbox_format: str | None = None,
) -> GroundingSample:
    image = row["image"]
    bbox = _coerce_bbox(_first_present(row, "bbox", "bounding_box", "box"), fmt=bbox_format)
    instruction = _first_present(row, "instruction", "query", "task", default="")
    target_type = _first_present(row, "data_type", "element_type", "target_type")
    ui_type = _first_present(row, "data_source", "platform", "application", "group")
    sample_id = _first_present(row, "id", "image_id", "img_filename", default=str(idx))

    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "image",
            "bbox",
            "bounding_box",
            "box",
            "instruction",
            "query",
            "task",
        }
    }

    return GroundingSample(
        image=image,
        instruction=str(instruction),
        bbox=bbox,
        image_size=image.size,
        benchmark=benchmark,
        target_type=str(target_type) if target_type else None,
        ui_type=str(ui_type) if ui_type else None,
        sample_id=str(sample_id),
        extra=extras,
    )


def _load_iter(
    repo_id: str,
    benchmark: str,
    split: str,
    *,
    streaming: bool = False,
    **load_kwargs: Any,
) -> Iterator[GroundingSample]:
    from datasets import load_dataset

    try:
        ds = load_dataset(repo_id, split=split, streaming=streaming, **load_kwargs)
        loaded_split = split
    except ValueError as exc:
        if split == "train" or "Unknown split" not in str(exc):
            raise
        # Some HF mirrors package benchmark examples as a single `train` split.
        # Treat that as the evaluation split rather than forcing every config to
        # know the mirror-specific naming convention.
        ds = load_dataset(repo_id, split="train", streaming=streaming, **load_kwargs)
        loaded_split = "train"
    for i, row in enumerate(ds):
        sample = _row_to_sample(row, benchmark, i)
        sample.extra.setdefault("requested_split", split)
        sample.extra.setdefault("loaded_split", loaded_split)
        yield sample


def _load_os_copilot_screenspot_v2(repo_id: str) -> Iterator[GroundingSample]:
    """Load the canonical OS-Copilot ScreenSpot-V2 JSON + image zip layout."""
    from huggingface_hub import hf_hub_download

    image_zip_path = hf_hub_download(repo_id, "screenspotv2_image.zip", repo_type="dataset")
    json_paths = [
        hf_hub_download(repo_id, filename, repo_type="dataset")
        for filename in _SCREENSPOT_V2_JSONS
    ]

    idx = 0
    with ZipFile(image_zip_path) as images:
        for json_path in json_paths:
            with open(json_path, encoding="utf-8") as f:
                rows = json.load(f)
            for row in rows:
                image_name = row["img_filename"]
                with images.open(f"screenspotv2_image/{image_name}") as image_file:
                    from PIL import Image

                    image = Image.open(BytesIO(image_file.read())).convert("RGB")
                sample = _row_to_sample(
                    {**row, "image": image},
                    "screenspot-v2",
                    idx,
                    bbox_format="xywh",
                )
                sample.extra.setdefault("requested_split", "test")
                sample.extra.setdefault("loaded_split", "os-copilot-json")
                yield sample
                idx += 1


def load_screenspot_v2(
    split: str = "test",
    *,
    repo_id: str = SCREENSPOT_V2_REPO,
    streaming: bool = False,
    **kwargs: Any,
) -> Iterator[GroundingSample]:
    """Yield ScreenSpot-V2 samples (1,272 in the test split)."""
    if repo_id == "OS-Copilot/ScreenSpot-v2":
        yield from _load_os_copilot_screenspot_v2(repo_id)
        return
    yield from _load_iter(repo_id, "screenspot-v2", split, streaming=streaming, **kwargs)


def _load_likaixin_screenspot_pro(repo_id: str) -> Iterator[GroundingSample]:
    """Load ScreenSpot-Pro from likaixin's per-domain JSON + PNG layout.

    `load_dataset` can't auto-detect this layout, so we `snapshot_download`
    the repo and walk `annotations/<domain>.json` + `images/<subdir>/*.png`.
    """
    from huggingface_hub import snapshot_download
    from PIL import Image

    local_root = Path(snapshot_download(repo_id, repo_type="dataset"))
    annotations_dir = local_root / "annotations"
    images_dir = local_root / "images"

    idx = 0
    skipped = 0
    for json_path in sorted(annotations_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            image_filename = row.get("img_filename")
            if not image_filename or row.get("bbox") is None:
                skipped += 1
                continue
            image_path = images_dir / image_filename
            if not image_path.exists():
                skipped += 1
                continue
            image = Image.open(image_path).convert("RGB")
            # Pro uses `ui_type` and `platform`; promote to the keys the unified
            # `_row_to_sample` already understands.
            normalized = dict(row)
            normalized.setdefault("data_type", normalized.pop("ui_type", None))
            normalized.setdefault("data_source", normalized.get("platform"))
            normalized["image"] = image
            sample = _row_to_sample(
                normalized, "screenspot-pro", idx, bbox_format="xyxy"
            )
            sample.extra.setdefault("application", row.get("application"))
            sample.extra.setdefault("group", row.get("group"))
            sample.extra.setdefault("annotation_file", json_path.name)
            yield sample
            idx += 1
    if skipped:
        warnings.warn(
            f"Skipped {skipped} invalid ScreenSpot-Pro rows with missing bbox/image.",
            stacklevel=2,
        )


def load_screenspot_pro(
    split: str = "test",
    *,
    repo_id: str = SCREENSPOT_PRO_REPO,
    streaming: bool = False,
    **kwargs: Any,
) -> Iterator[GroundingSample]:
    """Yield ScreenSpot-Pro samples (~1,581 high-resolution professional UIs)."""
    if repo_id == "likaixin/ScreenSpot-Pro":
        yield from _load_likaixin_screenspot_pro(repo_id)
        return
    yield from _load_iter(repo_id, "screenspot-pro", split, streaming=streaming, **kwargs)
