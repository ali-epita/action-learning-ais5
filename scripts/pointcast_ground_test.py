#!/usr/bin/env python3
"""Tune recipe target phrasings against your real screen.

Open the screen you want to test (e.g. System Settings on the General pane), then
ground a candidate phrase and see exactly where PointCast would click. Iterate on
the wording until the crosshair lands on the right control, then put that phrase
in the recipe.

    .venv-demo/bin/python scripts/pointcast_ground_test.py "General in the settings sidebar"
    .venv-demo/bin/python scripts/pointcast_ground_test.py --image shot.png "Storage"
    .venv-demo/bin/python scripts/pointcast_ground_test.py --list        # show built-in recipe targets
"""
from __future__ import annotations

import argparse
import sys

from PIL import Image, ImageDraw


def list_recipes() -> int:
    from ais5.pointcast.task.recipes import RECIPES

    for r in RECIPES:
        print(f"\n# {r.name}   (say: {', '.join(r.utterances[:3])} ...)")
        for i, s in enumerate(r.steps, 1):
            print(f"  step {i}: {s.target!r}")
        if r.question:
            print(f"  question: {r.question!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground a phrase against the live screen (recipe tuning)")
    ap.add_argument("phrase", nargs="?", help="the target phrase to ground")
    ap.add_argument("--model", default="mlx-qwen-r64-4bit")
    ap.add_argument("--image", default=None, help="ground against this screenshot instead of the live screen")
    ap.add_argument("--max-side", type=int, default=1512)
    ap.add_argument("--list", action="store_true", help="list built-in recipe targets and exit")
    a = ap.parse_args()

    if a.list:
        return list_recipes()
    if not a.phrase:
        ap.error("give a phrase to ground, or use --list")

    from ais5.pointcast.backends import get_backend

    if a.image:
        gimg = Image.open(a.image).convert("RGB")
        to_logical = None
    else:
        from ais5.pointcast.capture import capture_screen

        print("capturing the live screen (needs Screen Recording permission) ...")
        frame = capture_screen(a.max_side, 1)
        gimg = frame.image
        to_logical = frame.mapper.ground_to_logical

    print(f"loading {a.model} ...")
    backend = get_backend("mlx", model_path=a.model)
    backend.load()

    print(f'grounding: "{a.phrase}"')
    out = backend.predict(gimg, a.phrase)
    pt = out.parsed.point
    print(f"  raw    = {out.text!r}")
    if pt is None:
        print("  RESULT: no point parsed (the model did not locate it) — try rephrasing")
        return 1
    print(f"  point  = ({pt[0]:.0f}, {pt[1]:.0f})  in the {gimg.size[0]}x{gimg.size[1]} grounding image")
    if to_logical is not None:
        lx, ly = to_logical(*pt)
        print(f"  click  = ({lx:.0f}, {ly:.0f})  logical screen point (where it would click)")

    preview = gimg.copy()
    d = ImageDraw.Draw(preview)
    x, y = pt
    d.line([(x - 22, y), (x + 22, y)], fill="#22c55e", width=3)
    d.line([(x, y - 22), (x, y + 22)], fill="#22c55e", width=3)
    d.ellipse([x - 10, y - 10, x + 10, y + 10], outline="#22c55e", width=3)
    out_path = "/tmp/pointcast_ground_test.png"
    preview.save(out_path)
    print(f"  preview -> {out_path}   (open it to see where the crosshair landed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
