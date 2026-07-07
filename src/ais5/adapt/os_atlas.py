"""Streaming loader for OS-Copilot/OS-Atlas-data.

load_dataset only exposes the images (imagefolder, no labels). The grounding
labels live in per-domain JSON files that pair img_filename + instruction +
normalized bbox. This reads those JSONs, opens the matching image from the
domain zip, and yields rows in the {image, instruction, bbox} shape that
adapt_os_atlas_row consumes. bbox is denormalized from the dataset's [0, 1]
ratios to pixels.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from PIL import Image as PILImage

from ..utils.logging import get_logger

log = get_logger(__name__)

REPO_ID = "OS-Copilot/OS-Atlas-data"

# subset key -> (annotation json, [zip or split parts to concatenate])
_SUBSETS: dict[str, tuple[str, list[str]]] = {
    "uibert": ("mobile_domain/uibert_raw.json", ["mobile_domain/UIBert.zip"]),
    "ricosca": ("mobile_domain/ricosca.json", ["mobile_domain/rico_imgs.zip"]),
    "widget_captioning": ("mobile_domain/widget_captioning.json", ["mobile_domain/rico_imgs.zip"]),
    "aw_mobile": ("mobile_domain/aw_mobile.json", ["mobile_domain/mobile_images.zip"]),
    "amex": ("mobile_domain/amex_raw.json", [
        "mobile_domain/amex_images_part_aa",
        "mobile_domain/amex_images_part_ab",
        "mobile_domain/amex_images_part_ac",
    ]),
    "linux": ("desktop_domain/linux_splited.json", ["desktop_domain/linux_images.zip"]),
    "macos": ("desktop_domain/macos_splited.json", ["desktop_domain/macos_images.zip"]),
    "windows": ("desktop_domain/windows_splited.json", [
        "desktop_domain/windows_image_aa",
        "desktop_domain/windows_image_ab",
        "desktop_domain/windows_image_ac",
        "desktop_domain/windows_image_ad",
    ]),
    "seeclick_web": ("web_domain/seeclick_web.json", [
        "web_domain/seeclick_web_image_aa",
        "web_domain/seeclick_web_image_ab",
        "web_domain/seeclick_web_image_ac",
        "web_domain/seeclick_web_image_ad",
    ]),
}

# Single-zip mobile/desktop subsets; no multi-GB merge needed. fineweb (10 zips)
# and the split-zip subsets are supported but heavier.
DEFAULT_SUBSETS: tuple[str, ...] = ("uibert", "ricosca", "widget_captioning")


def available_subsets() -> list[str]:
    return sorted(_SUBSETS)


def _ensure_zip(repo_id: str, parts: list[str], cache_dir: str | None) -> str:
    from huggingface_hub import hf_hub_download

    local = [
        hf_hub_download(repo_id, p, repo_type="dataset", cache_dir=cache_dir)
        for p in parts
    ]
    if len(local) == 1:
        return local[0]
    merged = Path(local[0]).with_suffix(".merged.zip")
    if not merged.exists():
        # Write to a temp name and rename: an interrupted copy (preemption,
        # Ctrl-C, OOM kill) must not leave a truncated file that `exists()`
        # then treats as complete forever after.
        tmp = merged.with_suffix(".merged.zip.tmp")
        with open(tmp, "wb") as out:
            for part in local:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out)
        tmp.replace(merged)
    return str(merged)


def _zip_index(zf: ZipFile) -> dict[str, str]:
    idx: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        idx.setdefault(name, name)
        idx.setdefault(name.split("/")[-1], name)  # basename fallback
    return idx


def _resolve_name(img_filename: str, idx: dict[str, str]) -> str | None:
    if img_filename in idx:
        return idx[img_filename]
    base = img_filename.split("/")[-1]
    if base in idx:
        return idx[base]
    for name in idx.values():
        if name.endswith(img_filename):
            return name
    return None


def _iter_entries(data: Any) -> Iterator[tuple[str, list[dict]]]:
    """Yield (img_filename, elements) for both nested and flat JSON shapes."""
    rows = data if isinstance(data, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        img_filename = row.get("img_filename") or row.get("image")
        if not img_filename:
            continue
        elems = row.get("elements")
        if elems is None:
            if row.get("instruction") and row.get("bbox") is not None:
                yield img_filename, [row]
        elif isinstance(elems, list):
            yield img_filename, elems


def _to_pixel_bbox(bbox: Any, width: int, height: int) -> list[float] | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    # The dataset stores [0, 1] ratios; guard against a subset already in pixels.
    if max(x1, y1, x2, y2) <= 1.5:
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    return [x1, y1, x2, y2]


def stream_os_atlas(
    subsets: Iterable[str] | str = DEFAULT_SUBSETS,
    *,
    limit: int | None = None,
    repo_id: str = REPO_ID,
    cache_dir: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream OS-Atlas grounding rows as {image, instruction, bbox(pixels)} dicts.

    Each row is ready for `adapt_os_atlas_row`. One screenshot expands into one
    row per annotated element.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    n = 0
    for key in subsets:
        if key not in _SUBSETS:
            raise ValueError(
                f"Unknown OS-Atlas subset {key!r}; available: {available_subsets()}"
            )
        json_path, zip_parts = _SUBSETS[key]
        from huggingface_hub import hf_hub_download

        ann_path = hf_hub_download(repo_id, json_path, repo_type="dataset", cache_dir=cache_dir)
        zip_path = _ensure_zip(repo_id, zip_parts, cache_dir)
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        log.info("OS-Atlas subset %s: %d annotation entries", key, len(data))
        with ZipFile(zip_path) as zf:
            idx = _zip_index(zf)
            for img_filename, elems in _iter_entries(data):
                name = _resolve_name(img_filename, idx)
                if name is None:
                    continue
                try:
                    with zf.open(name) as fp:
                        img = PILImage.open(BytesIO(fp.read())).convert("RGB")
                    img.load()
                except Exception:
                    continue
                w, h = img.size
                for el in elems:
                    if not isinstance(el, dict):
                        continue
                    instruction = el.get("instruction")
                    bbox = _to_pixel_bbox(el.get("bbox"), w, h)
                    if not instruction or bbox is None:
                        continue
                    yield {"image": img, "instruction": str(instruction), "bbox": bbox}
                    n += 1
                    if limit is not None and n >= limit:
                        return


def os_atlas_dataset(
    subsets: Iterable[str] | str = DEFAULT_SUBSETS,
    *,
    limit: int | None = None,
    repo_id: str = REPO_ID,
    cache_dir: str | None = None,
):
    """Wrap `stream_os_atlas` as a torch IterableDataset for HF Trainer."""
    import torch

    spec = list(subsets) if not isinstance(subsets, str) else [subsets]

    class _OSAtlasIterable(torch.utils.data.IterableDataset):
        def __iter__(self) -> Iterator[dict[str, Any]]:
            return stream_os_atlas(spec, limit=limit, repo_id=repo_id, cache_dir=cache_dir)

    return _OSAtlasIterable()
