#!/usr/bin/env python3
"""M0 smoke test for the PointCast MLX backend.

Proves the on-device path end-to-end with no network:
  load mlx-qwen-r64-4bit -> predict -> parse_click  (the core contract)
  crop_then_click(...)                              (the Try-Twice substrate)
  peak_memory_bytes()                               (the trust-panel signal)

    .venv-demo/bin/python scripts/pointcast_smoke.py
    .venv-demo/bin/python scripts/pointcast_smoke.py --image some_ui.png --instruction "click the search box"
"""

from __future__ import annotations

import argparse
import time

from PIL import Image, ImageDraw

from ais5.pointcast.backends import get_backend
from ais5.tile.crop_then_click import CropConfig, crop_then_click


def synthetic_ui(w: int = 1280, h: int = 800) -> Image.Image:
    """A clean, unambiguous mock UI so grounding has a clear target offline."""
    img = Image.new("RGB", (w, h), "#f3f4f6")
    d = ImageDraw.Draw(img)
    # top bar
    d.rectangle([0, 0, w, 56], fill="#1f2937")
    d.text((20, 20), "MyApp", fill="#ffffff")
    d.text((w - 110, 20), "Settings", fill="#cbd5e1")
    # search box (clear target, centered in the top bar area)
    sx1, sy1, sx2, sy2 = w // 2 - 220, 14, w // 2 + 220, 44
    d.rectangle([sx1, sy1, sx2, sy2], fill="#ffffff", outline="#9ca3af", width=2)
    d.text((sx1 + 12, sy1 + 8), "Search", fill="#6b7280")
    # left sidebar
    d.rectangle([0, 56, 220, h], fill="#e5e7eb")
    for i, label in enumerate(["Home", "Files", "Shared", "Trash"]):
        d.text((24, 90 + i * 40), label, fill="#374151")
    # primary button bottom-right
    bx1, by1, bx2, by2 = w - 180, h - 80, w - 40, h - 36
    d.rectangle([bx1, by1, bx2, by2], fill="#2563eb")
    d.text((bx1 + 48, by1 + 14), "Send", fill="#ffffff")
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-qwen-r64-4bit")
    ap.add_argument("--image", default=None, help="optional UI screenshot; default = synthetic")
    ap.add_argument("--instruction", default="click the search box")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB") if args.image else synthetic_ui()
    if not args.image:
        img.save("/tmp/pointcast_smoke_ui.png")
        print("synthetic UI written to /tmp/pointcast_smoke_ui.png")

    print(f"\nloading backend ({args.model}) ...")
    t0 = time.perf_counter()
    backend = get_backend("mlx", model_path=args.model)
    backend.load()
    backend.reset_peak_memory()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # 1) single deterministic predict (benchmark-parity path)
    print(f'\n[1] predict  — instruction: "{args.instruction}"')
    t0 = time.perf_counter()
    out = backend.predict(img, args.instruction)
    dt = time.perf_counter() - t0
    print(f"    point   = {out.parsed.point}   (parser={out.parsed.parser})")
    print(f"    raw     = {out.text[:120]!r}")
    print(f"    latency = {dt:.2f}s   tokens={out.metadata.get('generation_tokens')}")

    # 2) crop_then_click composition (the Try-Twice substrate, free stability signals)
    print("\n[2] crop_then_click — two-stage zoom (768px window)")
    t0 = time.perf_counter()
    ctc = crop_then_click(backend, img, args.instruction, cfg=CropConfig(crop_size=768))
    dt = time.perf_counter() - t0
    print(f"    point        = {ctc.parsed.point}")
    print(f"    coarse_point = {ctc.metadata.get('coarse_point')}")
    print(f"    refined_pt   = {ctc.metadata.get('refined_point')}  frame={ctc.metadata.get('refined_coord_frame')}")
    print(f"    crop_box     = {ctc.metadata.get('crop_box')}")
    print(f"    latency      = {dt:.2f}s (2 calls)")

    # 3) trust-panel signal
    peak = backend.peak_memory_bytes()
    print(f"\n[3] peak memory = {peak / 1e9:.2f} GB" if peak else "\n[3] peak memory = (unavailable)")

    # sanity assertions for the pipeline (not accuracy)
    ok = out.parsed.point is not None and ctc.parsed.point is not None
    pw, ph = img.size
    if out.parsed.point is not None:
        x, y = out.parsed.point
        in_bounds = -5 <= x <= pw + 5 and -5 <= y <= ph + 5
    else:
        in_bounds = False
    print("\nRESULT:", "PASS" if (ok and in_bounds) else "CHECK (pipeline ran; review points above)")

    # draw a crosshair preview for eyeballing
    if out.parsed.point is not None:
        prev = img.copy()
        dd = ImageDraw.Draw(prev)
        x, y = out.parsed.point
        r = 22
        dd.ellipse([x - r, y - r, x + r, y + r], outline="#16a34a", width=5)
        dd.line([x - r - 12, y, x + r + 12, y], fill="#16a34a", width=3)
        dd.line([x, y - r - 12, x, y + r + 12], fill="#16a34a", width=3)
        prev.save("/tmp/pointcast_smoke_pred.png")
        print("crosshair preview -> /tmp/pointcast_smoke_pred.png")

    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
