#!/usr/bin/env python3
"""Generate the Task-3 (efficiency / quantization) deep-dive notebook.

Pure analysis of the cached quant grid (no GPU): the accuracy/VRAM/latency
trade-off, VRAM reduction, the encoder-vs-decoder latency split that explains
why weight-only quant is *slower*, a deployability frontier, and cross-benchmark
quant stability. Self-contained (no make_deck import → inline backend works).

    python scripts/build_task3_deepdive.py <out.ipynb> [run_id]
"""

from __future__ import annotations

import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "notebooks/AIS5_Task3_DeepDive.ipynb"
RUN_ID = sys.argv[2] if len(sys.argv) > 2 else "2026-06-08T18-44-31"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(
f"""# Task 3 — Efficiency & quantization deep-dive

**Owner: Ali Cherri · H3.** Beyond the headline trade-off, this notebook digs into *where* the cost
goes: the VRAM reduction per model, the **encoder-vs-decoder latency split** that explains why
weight-only quant is slower at batch-1, an on-device **deployability frontier**, and how stable the
quant ranking is across benchmarks. All from the cached grid (`{RUN_ID}`) — no GPU needed.

**H3 restated:** low-bit quantization should make small VLMs deployable on-device (≤ 8 GB) cheaply.
**Finding:** it's a **memory** win (−48% to −69% VRAM at ≤ 3 pp accuracy loss), **not** a latency win
— the LLM-decode stage *grows* under weight-only quant (dequant overhead)."""))

cells.append(code(
f'''import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import ais5
REPO_ROOT = Path(ais5.__file__).resolve().parents[2]
RUN = REPO_ROOT / "results" / "{RUN_ID}"

plt.rcParams.update({{
    "font.family": "DejaVu Sans", "font.size": 12.5, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#eef2f7", "axes.edgecolor": "#cbd5e1", "figure.facecolor": "white", "legend.frameon": False,
}})
QC = {{"fp16": "#94a3b8", "bnb-8bit": "#2563eb", "bnb-4bit-nf4": "#16a34a"}}
QLAB = {{"fp16": "FP16", "bnb-8bit": "INT8", "bnb-4bit-nf4": "NF4 4-bit"}}
QORD = {{"fp16": 0, "bnb-8bit": 1, "bnb-4bit-nf4": 2}}
SHORT = {{"Qwen2.5-VL-3B-Instruct": "Qwen-3B", "paligemma-3b-mix-448": "PaliGemma-3B", "ShowUI-2B": "ShowUI-2B"}}

seen = {{}}
for line in open(RUN / "task3" / "task3_grid.jsonl"):
    if line.strip():
        r = json.loads(line)
        if r.get("n_samples", 0) > 0:
            seen[(r["model"], r["quant"], r["benchmark"])] = r
df = pd.DataFrame(seen.values())
df["model_s"] = df["model"].map(SHORT).fillna(df["model"])
df["qlab"] = df["quant"].map(QLAB)
df["qord"] = df["quant"].map(QORD)
df["acc_pct"] = df["accuracy"] * 100
df["encode_ms"] = df["components_visual_encode_mean_ms"]
df["decode_ms"] = df["components_llm_decode_mean_ms"]
print(f"loaded grid: {{df['model'].nunique()}} models x {{df['quant'].nunique()}} quants x {{df['benchmark'].nunique()}} benchmarks = {{len(df)}} cells")
v2 = df[df.benchmark == "screenspot-v2"].sort_values(["model_s", "qord"])
'''))

cells.append(md("""## 1. The trade-off table (ScreenSpot-V2)

Accuracy, peak VRAM, end-to-end latency, and the encode/decode split for every model × quant level."""))
cells.append(code(
'''tbl = v2[["model_s", "qlab", "acc_pct", "peak_vram_gb", "latency_mean_ms", "encode_ms", "decode_ms"]].copy()
tbl.columns = ["model", "quant", "acc %", "VRAM GB", "latency ms", "encode ms", "decode ms"]
tbl = tbl.round({"acc %": 1, "VRAM GB": 2, "latency ms": 0, "encode ms": 0, "decode ms": 0}).reset_index(drop=True)
display(tbl)
'''))

cells.append(md("""## 2. The memory win — VRAM by quant level

4-bit roughly halves peak VRAM. Every 3–4 B model drops far under the 8 GB on-device budget."""))
cells.append(code(
'''fig, ax = plt.subplots(figsize=(10, 4.6))
models = list(v2.model_s.unique()); quants = ["fp16", "bnb-8bit", "bnb-4bit-nf4"]
x = np.arange(len(models)); w = 0.26
for i, q in enumerate(quants):
    vals = [v2[(v2.model_s == m) & (v2.quant == q)]["peak_vram_gb"].mean() for m in models]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=QLAB[q], color=QC[q])
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.12, f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(8, ls="--", lw=1.4, color="#dc2626"); ax.text(len(models) - 0.5, 8.2, "8 GB budget", color="#dc2626", fontsize=10, ha="right")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("peak VRAM (GB)")
ax.set_title("4-bit ~halves VRAM — all fit under 8 GB"); ax.legend()
for m in models:
    f16 = v2[(v2.model_s == m) & (v2.quant == "fp16")]["peak_vram_gb"].mean()
    nf4 = v2[(v2.model_s == m) & (v2.quant == "bnb-4bit-nf4")]["peak_vram_gb"].mean()
    xi = models.index(m)
    ax.annotate(f"−{(1-nf4/f16)*100:.0f}%", (xi + w, nf4), textcoords="offset points", xytext=(0, 16),
                ha="center", color="#16a34a", fontweight="bold", fontsize=10)
fig.tight_layout(); plt.show()
'''))

cells.append(md(
"""## 3. Why quant is *slower* — the encoder-vs-decoder split ⭐

The signature insight. Each bar splits end-to-end latency into **vision-encode** (bottom) and
**LLM-decode** (top). Vision-encode is ~constant across quant levels, but **LLM-decode balloons**
under weight-only INT8/NF4 — the dequantize-on-the-fly cost in the autoregressive decode dominates at
batch-1. That's why H3's "−35% latency" doesn't hold for weight-only quant."""))
cells.append(code(
'''fig, ax = plt.subplots(figsize=(11, 4.8))
cats, enc, dec = [], [], []
for m in v2.model_s.unique():
    for q in ["fp16", "bnb-8bit", "bnb-4bit-nf4"]:
        row = v2[(v2.model_s == m) & (v2.quant == q)]
        if len(row):
            cats.append(f"{m}\\n{QLAB[q]}"); enc.append(row.encode_ms.mean()); dec.append(row.decode_ms.mean())
x = np.arange(len(cats))
ax.bar(x, enc, 0.62, label="vision encode", color="#0d9488")
b = ax.bar(x, dec, 0.62, bottom=enc, label="LLM decode", color="#f59e0b")
for i, (e, d) in enumerate(zip(enc, dec)):
    ax.text(i, e + d + 30, f"{e+d:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9); ax.set_ylabel("mean latency (ms / sample)")
ax.set_title("End-to-end latency = vision-encode + LLM-decode  (decode balloons under quant)"); ax.legend()
fig.tight_layout(); plt.show()
'''))

cells.append(md("""## 4. Deployability frontier (ScreenSpot-V2)

Each point is a (model, quant) config: VRAM on x, accuracy on y. Everything left of the 8 GB line is
on-device-deployable. The quantized generalist/specialist configs cluster in the **bottom-left → top
sweet spot** (small footprint, competitive accuracy)."""))
cells.append(code(
'''fig, ax = plt.subplots(figsize=(9.5, 5.2))
mk = {"fp16": "o", "bnb-8bit": "s", "bnb-4bit-nf4": "D"}
mcol = {"Qwen-3B": "#2563eb", "PaliGemma-3B": "#0d9488", "ShowUI-2B": "#f59e0b"}
for _, r in v2.iterrows():
    if r.acc_pct < 1: continue  # skip PaliGemma's 0% (uninformative on this axis)
    ax.scatter(r.peak_vram_gb, r.acc_pct, s=170, marker=mk[r.quant],
               color=mcol.get(r.model_s, "#7c3aed"), edgecolor="white", linewidth=1.4, zorder=3)
    ax.annotate(f"{r.model_s} {QLAB[r.quant]}", (r.peak_vram_gb, r.acc_pct),
                textcoords="offset points", xytext=(7, 6), fontsize=8.5, color="#33475f")
ax.axvline(8, ls="--", lw=1.4, color="#dc2626"); ax.text(8.1, ax.get_ylim()[0] + 2, "8 GB", color="#dc2626", fontsize=10)
ax.set_xlabel("peak VRAM (GB)"); ax.set_ylabel("ScreenSpot-V2 accuracy (%)")
ax.set_title("Deployability frontier — left of 8 GB = on-device")
handles = [Patch(color=c, label=m) for m, c in mcol.items() if m != "PaliGemma-3B"]
ax.legend(handles=handles, loc="lower right")
fig.tight_layout(); plt.show()
'''))

cells.append(md("""## 5. Is quant degradation stable across benchmarks?

Accuracy drop from FP16 → NF4 (4-bit), per benchmark. If the bars are small everywhere, 4-bit is a
safe default; a large drop on a harder benchmark would warn against it there."""))
cells.append(code(
'''benches = ["screenspot-v2", "screenspot-pro", "osworld-g"]
fig, ax = plt.subplots(figsize=(10, 4.4))
models = [m for m in df.model_s.unique() if m != "PaliGemma-3B"]
x = np.arange(len(benches)); w = 0.36
for i, m in enumerate(models):
    drops = []
    for bch in benches:
        f16 = df[(df.model_s == m) & (df.quant == "fp16") & (df.benchmark == bch)]["acc_pct"]
        nf4 = df[(df.model_s == m) & (df.quant == "bnb-4bit-nf4") & (df.benchmark == bch)]["acc_pct"]
        drops.append((f16.values[0] - nf4.values[0]) if len(f16) and len(nf4) else np.nan)
    bars = ax.bar(x + (i - 0.5) * w, drops, w, label=m, color=["#2563eb", "#f59e0b"][i % 2])
    for b, d in zip(bars, drops):
        if not np.isnan(d): ax.text(b.get_x() + b.get_width() / 2, d + 0.1, f"{d:+.1f}", ha="center", fontsize=9)
ax.axhline(3, ls=":", color="#dc2626", lw=1.2); ax.text(0, 3.1, "3 pp", color="#dc2626", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([b.replace("screenspot-", "SS-") for b in benches]); ax.set_ylabel("FP16 → NF4 accuracy drop (pp)")
ax.set_title("4-bit accuracy cost by benchmark (lower = safer)"); ax.legend()
fig.tight_layout(); plt.show()
'''))

cells.append(md(
"""## Takeaways

1. **Memory win is large and consistent:** NF4 cuts peak VRAM −48% (ShowUI) to −69% (PaliGemma);
   every model lands well under 8 GB → genuinely on-device-deployable.
2. **No latency win at batch-1:** the encode/decode split shows the **LLM-decode stage grows** under
   weight-only INT8/NF4 (dequant overhead), so end-to-end latency goes *up*, not down. INT8 is worst.
3. **Accuracy cost is small and stable:** FP16 → NF4 drop stays within a few pp across benchmarks →
   4-bit is a safe default for deployment.
4. **Practical guidance:** for on-device grounding, ship **NF4 4-bit** for the memory savings; if
   latency matters, weight-only quant is the wrong tool (need real low-bit *compute* / batching).

*Caveat: weight-only NF4 (activations stay bf16) ≠ literal W4A8; latency is A100 batch-1 wall-clock,
relative-within-grid. Preview-scale subsets — trends stable, absolutes tighten at full scale.*"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, f"({len(cells)} cells)")
