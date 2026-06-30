#!/usr/bin/env python3
"""Calibrate the Try-Twice stability gate on held-out ScreenSpot-V2 (local MLX).

For each sample it runs the FULL crop ladder without auto-accepting (so every
stage's agreement displacement is recorded), then sweeps the agreement threshold
``displacement_frac`` and reports, per threshold:

  coverage  = fraction of samples the gate would auto-accept
  precision = of those, fraction that actually land in the gold box
  committed-accuracy = precision (what the user experiences when it auto-clicks)

Pick the largest ``displacement_frac`` whose precision clears your bar (e.g.
>=0.95) — that maximizes hands-free auto-clicks without silent mis-clicks.
Everything below that bar falls through to refuse-and-ask / disambiguation.

Heavy + offline-after-download: needs HF access for the dataset, the MLX model,
and ~N×(1+stages) model calls (~9s each). Start small:

    .venv-demo/bin/python scripts/pointcast_calibrate_gate.py --limit 20
    .venv-demo/bin/python scripts/pointcast_calibrate_gate.py --limit 200 --crop-sizes 768,512
"""

from __future__ import annotations

import argparse
import time

CROP_RE_PREFIX = "crop"


def collect(backend, sample, ground_max_side, crop_sizes):
    from ais5.pointcast.geometry import compute_ground_scale
    from ais5.tile.try_twice import TryTwiceConfig, try_twice

    orig = sample.image.convert("RGB")
    ow, oh = orig.size
    f = compute_ground_scale(ow, oh, ground_max_side)
    gimg = orig.resize((max(1, round(ow * f)), max(1, round(oh * f)))) if f < 1.0 else orig

    # displacement_frac=-1 => never auto-accept => the full ladder runs and every
    # stage displacement is recorded for post-hoc threshold sweeping.
    cfg = TryTwiceConfig(crop_sizes=crop_sizes, displacement_frac=-1.0)
    r = try_twice(backend, gimg, sample.instruction, cfg)

    def to_orig(p):
        return (p[0] / f, p[1] / f) if p else None

    coarse = r.stages[0].point
    stages = []
    for st in r.stages[1:]:
        if st.point is None or st.displacement is None:
            continue
        crop = int(st.name.replace(CROP_RE_PREFIX, ""))
        stages.append({"crop": crop, "disp": st.displacement, "point": to_orig(st.point)})
    return {"bbox": sample.bbox, "coarse": to_orig(coarse), "stages": stages}


def evaluate(records, frac, point_in_bbox):
    n = len(records)
    committed = correct = 0
    for rec in records:
        accepted = None
        for s in rec["stages"]:
            if s["disp"] <= frac * s["crop"]:
                accepted = s["point"]
                break
        if accepted is not None:
            committed += 1
            if point_in_bbox(accepted, rec["bbox"]):
                correct += 1
    coverage = committed / n if n else 0.0
    precision = correct / committed if committed else 0.0
    return coverage, precision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-qwen-r64-4bit")
    ap.add_argument("--benchmark", default="screenspot-v2")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ground-max-side", type=int, default=1512)
    ap.add_argument("--crop-sizes", default="768,512")
    ap.add_argument("--precision-target", type=float, default=0.95)
    a = ap.parse_args()

    crop_sizes = tuple(int(x) for x in a.crop_sizes.split(","))

    from PIL import Image

    from ais5.data.registry import iter_benchmark
    from ais5.eval.click import point_in_bbox
    from ais5.pointcast.backends import get_backend

    print(f"loading {a.model} ...")
    backend = get_backend("mlx", model_path=a.model)
    backend.load()
    backend.predict(Image.new("RGB", (96, 96), "white"), "warmup")  # warm Metal kernels

    records = []
    t0 = time.perf_counter()
    for i, sample in enumerate(iter_benchmark(a.benchmark, limit=a.limit)):
        try:
            records.append(collect(backend, sample, a.ground_max_side, crop_sizes))
        except Exception as e:  # noqa: BLE001
            print(f"  sample {i} skipped: {e!r}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.limit}  ({(time.perf_counter() - t0):.0f}s)")

    if not records:
        print("no samples collected (dataset/model unavailable?)")
        return 1

    # baselines
    coarse_acc = sum(1 for r in records if r["coarse"] and point_in_bbox(r["coarse"], r["bbox"])) / len(records)
    final_acc = sum(
        1 for r in records if r["stages"] and point_in_bbox(r["stages"][-1]["point"], r["bbox"])
    ) / len(records)

    print(f"\nn={len(records)}  coarse-only acc={coarse_acc:.3f}  full-ladder-final acc={final_acc:.3f}\n")
    print(f"  {'frac':>6}  {'coverage':>9}  {'precision':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}")
    best = None
    for frac in [round(0.01 * k, 2) for k in range(1, 21)]:
        cov, prec = evaluate(records, frac, point_in_bbox)
        star = ""
        if prec >= a.precision_target and (best is None or cov > best[1]):
            best = (frac, cov, prec)
            star = "  ←"
        print(f"  {frac:>6.2f}  {cov:>9.3f}  {prec:>9.3f}{star}")

    if best:
        print(
            f"\nRecommended: gate_displacement_frac={best[0]} "
            f"(coverage {best[1]:.0%} at precision {best[2]:.0%} ≥ {a.precision_target:.0%})"
        )
    else:
        print(f"\nNo threshold reached precision {a.precision_target:.0%} — default to N=1 + always-confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
