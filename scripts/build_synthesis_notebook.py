#!/usr/bin/env python3
"""Generate the cross-task synthesis notebook (H1 + H2 + H3 in one story).

Loads cached results from a completed run and renders the integrated charts
(reusing scripts/make_deck.py's chart builders) + a verdict scorecard + a
qualitative gallery (zero-shot vs LoRA vs specialist). Emits an unexecuted
.ipynb to run with `nbconvert --execute`.

    python scripts/build_synthesis_notebook.py <out.ipynb> [run_id]
"""

from __future__ import annotations

import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "notebooks/AIS5_Synthesis.ipynb"
RUN_ID = sys.argv[2] if len(sys.argv) > 2 else "2026-06-08T18-44-31"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(
f"""# AIS 5 — Small VLMs vs. GUI Specialists: cross-task synthesis

**Modeling defense · all three hypotheses in one story.**

> **Central question:** under an on-device budget (**≤ 4 B params, ≤ 8 GB VRAM**), can a small
> *generalist* VLM — LoRA-adapted **(H1)** and 4-bit-quantized **(H3)** — match or beat specialized
> GUI models on screen grounding, with crop-then-click **(H2)** recovering 4K small targets?

**Verdict (spoiler):** yes. An adapted Qwen2.5-VL-3B reaches **85.5%** on ScreenSpot-V2 (beating the
ShowUI specialist's 72.5%), 4-bit quantization **halves its VRAM to ~3 GB** at ≤ 3 pp accuracy loss,
and crop-then-click adds **+12 pp** on 4K ScreenSpot-Pro. The charts below are built from the cached
run `{RUN_ID}`.
"""))

cells.append(code(
f'''import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from IPython.display import Image, display

import ais5
REPO_ROOT = Path(ais5.__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_deck as deck  # tested chart builders (baselines / pareto / quant / cross-benchmark / task1 / task2)
# make_deck switches matplotlib to the Agg backend on import; re-enable inline so
# our own plt figures still render in the notebook (the make_deck charts savefig + display as files).
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# Polished, cohesive theme for every chart.
deck.COLORS.update({{"Qwen2.5-VL-3B-Instruct": "#2563eb", "paligemma-3b-mix-448": "#0d9488",
                     "ShowUI-2B": "#f59e0b", "OS-Atlas-Pro-4B": "#7c3aed"}})
plt.rcParams.update({{
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef2f7", "axes.edgecolor": "#cbd5e1", "figure.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=["#2563eb", "#0d9488", "#f59e0b", "#7c3aed", "#dc2626"]),
}})

RUN = REPO_ROOT / "results" / "{RUN_ID}"
FIG = RUN / "figures_synth"; FIG.mkdir(parents=True, exist_ok=True)
grid = deck.load_grid(RUN)
base = deck.load_baselines(RUN)
print(f"loaded {{len(grid)}} quant-grid cells, {{len(base)}} baselines from {{RUN.name}}")

def show(path):
    if path and Path(path).exists():
        display(Image(filename=str(path)))
    else:
        print("(chart unavailable)")
'''))

cells.append(md("""## Verdict scorecard

Three hypotheses, three outcomes — the headline before the evidence."""))

cells.append(code(
'''fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.3))
cards = [
    ("H1 · LoRA adaptation", "✅ exceeded", "85.5%", "adapted Qwen beats ShowUI (72.5%)", "#16a34a"),
    ("H2 · crop-then-click", "✅ confirmed", "+12 pp", "recovers 4K small targets", "#16a34a"),
    ("H3 · 4-bit quantization", "⚠ split", "−60% VRAM", "halves VRAM, but no latency win", "#d97706"),
]
for ax, (h, v, big, sub, col) in zip(axes, cards):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.92, transform=ax.transAxes,
                               facecolor="#f5f8fd", edgecolor=col, linewidth=2.4))
    ax.text(0.5, 0.84, h, ha="center", fontsize=12.5, fontweight="bold", color="#0f2540", transform=ax.transAxes)
    ax.text(0.5, 0.66, v, ha="center", fontsize=11.5, color=col, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, big, ha="center", fontsize=30, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.16, sub, ha="center", fontsize=10, color="#33475f", transform=ax.transAxes)
fig.suptitle("Verdict — three hypotheses", fontsize=15, fontweight="bold")
fig.tight_layout(); display(fig); plt.close(fig)
'''))

cells.append(md("""## Starting point — zero-shot baselines (ScreenSpot-V2)

Generalists start strong zero-shot (well above the proposal's ~27% guess); the **ShowUI specialist
leads**, which frames H1. PaliGemma emits text/refusals rather than coordinates → needs adaptation."""))
cells.append(code('show(deck.chart_baselines(base, FIG)) if base else print("no baselines")'))

cells.append(md("""## H1 — LoRA adaptation closes (and beats) the gap

Adapting Qwen with LoRA lifts ScreenSpot-V2 from 61.5% (zero-shot) to **81–85.5%** across ranks,
**beating the ShowUI specialist (72.5%) at every rank**. Rank effect is backbone-dependent: Qwen
saturates early, PaliGemma only adapts at high rank (0.5% → 42% from r8 → r64)."""))
cells.append(code('show(deck.chart_task1(RUN, FIG, base))'))

cells.append(md("""## H2 — crop-then-click recovers 4K small targets

On 4K ScreenSpot-Pro, plain resolution scaling doesn't help (2× even hurts). **Crop-then-click
recovers +6 pp (512) to +12 pp (768)** vs the ~4% base — zoom-and-refine is the lever, at ~2× latency."""))
cells.append(code('show(deck.chart_task2(RUN, FIG))'))

cells.append(md("""## H3 — quantization for deployability (the owner's task)

The on-device win is **memory, not speed**. 4-bit (NF4) roughly **halves peak VRAM** (Qwen 7.85 → 3.10 GB)
at ≤ 3 pp accuracy loss — every 3–4 B model fits far under the 8 GB budget. But weight-only quant is
**not faster** at batch-1 (memory-bound; INT8 is slowest), so H3's “−35% latency” is refuted."""))
cells.append(code(
'''show(deck.chart_pareto(grid, "screenspot-v2", "peak_vram_gb", "peak VRAM (GB)", FIG, "synth_vram.png", vline=8))
show(deck.chart_pareto(grid, "screenspot-v2", "latency_mean_ms", "mean latency (ms / sample)", FIG, "synth_lat.png"))
show(deck.chart_quant_effect(grid, "screenspot-v2", FIG))
'''))

cells.append(md("""## No single winner across benchmarks

A difficulty ladder (V2 ≫ OSWorld-G > Pro) and a split podium: the **specialist tops V2**, but the
**generalist tops OSWorld-G**. Quant ranking stays stable across all three benchmarks."""))
cells.append(code('show(deck.chart_cross_benchmark(grid, FIG))'))

cells.append(md(
"""## The synthesis

Stack the levers and the thesis lands: take the **small generalist**, **LoRA-adapt it (H1)** so it
beats the specialist, then **4-bit-quantize it (H3)** so it fits in **~3 GB**. The result is an
on-device-budget model (≤ 4 B, ≤ 8 GB) that **out-grounds the specialist** — with crop-then-click
**(H2)** available as a zoom-and-refine policy for the hardest 4K targets."""))
cells.append(code(
'''labels = ["ShowUI-2B\\n(specialist, FP16)", "Qwen-3B\\nzero-shot", "Qwen-3B\\n+LoRA (H1)", "Qwen-3B\\n+LoRA +4bit (H1+H3)"]
acc = [72.5, 61.5, 85.5, 84.0]            # V2 accuracy (%)
vram = [5.57, 7.85, 7.85, 3.10]           # peak VRAM (GB)
colors = ["#f59e0b", "#94a3b8", "#2563eb", "#16a34a"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
b1 = a1.bar(labels, acc, color=colors)
a1.axhline(72.5, ls="--", lw=1.3, color="#f59e0b", alpha=0.8)
a1.set_ylabel("ScreenSpot-V2 accuracy (%)"); a1.set_title("Accuracy — adapted generalist wins"); a1.set_ylim(0, 100)
for r, v in zip(b1, acc): a1.text(r.get_x()+r.get_width()/2, v+1.5, f"{v:.1f}", ha="center", fontweight="bold", fontsize=11)
a1.tick_params(axis="x", labelsize=9)
b2 = a2.bar(labels, vram, color=colors)
a2.axhline(8, ls="--", lw=1.3, color="#dc2626", alpha=0.8); a2.text(0, 8.2, "8 GB on-device budget", color="#dc2626", fontsize=9)
a2.set_ylabel("peak VRAM (GB)"); a2.set_title("Footprint — quantized fits in ~3 GB"); a2.set_ylim(0, 12)
for r, v in zip(b2, vram): a2.text(r.get_x()+r.get_width()/2, v+0.2, f"{v:.2f}", ha="center", fontweight="bold", fontsize=11)
a2.tick_params(axis="x", labelsize=9)
fig.suptitle("Small adapted + quantized generalist vs. the specialist", fontsize=15, fontweight="bold")
fig.tight_layout(); display(fig); plt.close(fig)
'''))

cells.append(md(
"""## Qualitative gallery — what the models actually click

Green box = gold target, dot = predicted click (green = hit, red = miss). Compare the **zero-shot**
generalist, the **LoRA-adapted** generalist, and the **specialist** on the same ScreenSpot-V2 screens."""))
cells.append(code(
'''from ais5.viz import plot_predictions
from ais5.models import get_model
from ais5.data import iter_benchmark

samples = list(iter_benchmark("screenspot-v2", limit=6))
panels = [
    ("qwen2.5-vl-3b", None, "Qwen2.5-VL-3B — zero-shot"),
    ("qwen2.5-vl-3b", str(REPO_ROOT / "checkpoints" / "qwen2.5-vl-3b-lora-r32"), "Qwen2.5-VL-3B + LoRA (r=32)"),
    ("showui-2b", None, "ShowUI-2B — specialist"),
]
for base_id, adapter, title in panels:
    try:
        m = get_model(base_id, **({"peft_adapter": adapter} if adapter else {}))
        display(plot_predictions(m, samples, title=title)); plt.close("all")
        del m
        import gc, torch; gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()
    except Exception as e:
        print(f"skip {title}: {type(e).__name__}: {e}")
'''))

cells.append(md(
"""## Takeaways

1. **H1 (confirmed, exceeded):** LoRA adaptation makes a 3 B generalist *beat* a GUI specialist on
   ScreenSpot-V2 (85.5% vs 72.5%). The earlier failure was a coordinate-convention bug (UGround
   [0,1000] targets treated as pixels), not a modeling limit.
2. **H2 (confirmed):** crop-then-click recovers +6–+12 pp on 4K small targets; brute-force
   resolution scaling does not.
3. **H3 (split):** 4-bit quantization is a **memory** win (−60% VRAM, fits ≤ 8 GB) at ≤ 3 pp
   accuracy loss — but **not** a latency win at batch-1.
4. **Together:** an adapted, 4-bit generalist runs in **~3 GB** and out-grounds the specialist — the
   on-device thesis holds.

*Numbers are from a preview-scale run; absolute percentages tighten at full scale, trends are stable.*"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, f"({len(cells)} cells)")
