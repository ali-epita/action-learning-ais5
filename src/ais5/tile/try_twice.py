"""Self-correcting Try-Twice grounding harness (Task 2 centerpiece).

Wraps any ``GUIModel`` in a verify-and-retry loop:

  1. coarse deterministic predict on the full image → coarse point
  2. crop-then-reclick at progressively tighter windows (default 768 then 512),
     each centered on the current best estimate, mapped back to full coords
  3. a stability gate ACCEPTS as soon as a re-click AGREES with the prior point
     (displacement ≤ ``displacement_frac × crop``). If the ladder never agrees,
     the result is returned **unaccepted** ("uncertain") so the caller can
     refuse-and-ask / disambiguate instead of committing a silent mis-click.

Deterministic (do_sample=False) — the benchmarked path. Reuses
``crop_then_click`` clamping so geometry matches the measured policy. The gate
uses only free signals (parse success + cross-stage agreement) — there are no
logits/confidence from ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from PIL.Image import Image

from ..models.base import GUIModel
from .crop_then_click import _clamp_crop_box

Point = tuple[float, float]


@dataclass
class TryTwiceConfig:
    crop_sizes: tuple[int, ...] = (768, 512)  # ladder, widest → tightest
    displacement_frac: float = 0.06  # agreement threshold as a fraction of the crop size
    min_crop: int = 384  # never crop below this (element-scale floor; avoids over-zooming)


@dataclass
class TryTwiceStage:
    name: str
    point: Point | None
    crop_box: tuple[int, int, int, int] | None = None
    displacement: float | None = None
    threshold: float | None = None
    accepted: bool = False


@dataclass
class TryTwiceResult:
    point: Point | None  # best estimate in full-image pixel coords
    accepted: bool  # did the stability gate lock it?
    badge: str  # short status for the HUD
    stages: list[TryTwiceStage] = field(default_factory=list)
    candidates: list[Point] = field(default_factory=list)  # distinct points seen (for disambiguation)
    metadata: dict[str, Any] = field(default_factory=dict)


def _dedup(points: list[Point], tol: float = 8.0) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if p is None:
            continue
        if all(hypot(p[0] - q[0], p[1] - q[1]) > tol for q in out):
            out.append((round(p[0], 1), round(p[1], 1)))
    return out


def _reclick(
    model: GUIModel,
    image: Image,
    instruction: str,
    center: Point,
    crop_size: int,
    hires: tuple[Image, float] | None = None,
):
    """One crop-and-repredict pass. Returns (point in BASE-image coords, box).

    With ``hires=(full_image, scale)`` — ``scale`` = base_px / full_px, < 1.0 —
    the crop is taken from the FULL-resolution capture instead of the (possibly
    downscaled) base image: the model sees ``crop_size`` NATIVE pixels, the
    exact condition H2 measured (+6/+12 pp on small targets), and the refined
    point is mapped back into base coordinates. The returned box is in the
    coordinates of the image that was cropped.
    """
    if hires is not None:
        full, f = hires
        if f and f < 1.0:
            fw, fh = full.size
            box = _clamp_crop_box(center[0] / f, center[1] / f, crop_size, fw, fh)
            x1, y1, _, _ = box
            crop = full.crop(box)
            out = model.predict(crop, instruction)
            pt = out.parsed.point
            if pt is None:
                return None, box
            rx, ry = pt
            cw, ch = crop.size
            if 0 <= rx <= cw and 0 <= ry <= ch:
                # crop-local → full-image px → base (grounding) coords
                return ((rx + x1) * f, (ry + y1) * f), box
            return (rx * f, ry * f), box  # full-frame answer → base coords

    w, h = image.size
    box = _clamp_crop_box(center[0], center[1], crop_size, w, h)
    x1, y1, _, _ = box
    crop = image.crop(box)
    out = model.predict(crop, instruction)
    pt = out.parsed.point
    if pt is None:
        return None, box
    rx, ry = pt
    cw, ch = crop.size
    if 0 <= rx <= cw and 0 <= ry <= ch:
        return (rx + x1, ry + y1), box  # crop-local → full-image
    return (rx, ry), box  # some VLMs answer in full-image coords even on a crop


def try_twice(
    model: GUIModel,
    image: Image,
    instruction: str,
    cfg: TryTwiceConfig | None = None,
    *,
    hires: tuple[Image, float] | None = None,
) -> TryTwiceResult:
    """``hires=(full_resolution_image, scale)`` upgrades the re-click stages to
    crop the native-resolution capture instead of the downscaled ``image``
    (scale = image_px / full_px). All returned coordinates stay in ``image``
    (base) space, so callers map points back exactly as before."""
    cfg = cfg or TryTwiceConfig()
    stages: list[TryTwiceStage] = []
    candidates: list[Point] = []
    use_hires = hires is not None and bool(hires[1]) and hires[1] < 1.0

    coarse = model.predict(image, instruction)
    cpt = coarse.parsed.point
    stages.append(TryTwiceStage("coarse", cpt))
    if cpt is None:
        return TryTwiceResult(None, False, "no match", stages, [], {"reason": "stage1_parse_fail", "calls": 1})

    candidates.append(cpt)
    best = cpt
    for crop_size in cfg.crop_sizes:
        cs = max(int(crop_size), cfg.min_crop)
        rpt, box = _reclick(model, image, instruction, best, cs, hires if use_hires else None)
        if rpt is None:
            stages.append(TryTwiceStage(f"crop{cs}", None, box))
            continue  # crop parse failed — keep current best, try the next window
        disp = hypot(rpt[0] - best[0], rpt[1] - best[1])
        # Displacement is measured in base coords; in hires mode a cs-px native
        # crop only spans cs*scale base px, so scale the window accordingly to
        # keep the same fraction-of-window agreement semantics.
        win_base = cs * hires[1] if use_hires else cs
        thr = cfg.displacement_frac * win_base
        accepted = disp <= thr
        stages.append(TryTwiceStage(f"crop{cs}", rpt, box, disp, thr, accepted))
        candidates.append(rpt)
        best = rpt
        if accepted:
            return TryTwiceResult(
                rpt, True, f"locked ({cs})", stages, _dedup(candidates),
                {"calls": len(stages), "displacement": disp, "threshold": thr},
            )

    # ladder exhausted without agreement → uncertain (caller refuses/disambiguates)
    return TryTwiceResult(
        best, False, "uncertain", stages, _dedup(candidates),
        {"reason": "ladder_exhausted", "calls": len(stages)},
    )
