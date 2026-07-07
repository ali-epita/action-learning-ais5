"""Qualitative visualization: draw a model's predicted click + the gold target.

    from ais5.viz import plot_predictions
    plot_predictions(model, samples, title="...")

Green box = gold target bbox; dot = predicted click (green if it lands inside
the box, red if it misses); ✓/✗ + the instruction as each subplot title.
"""

from __future__ import annotations

from typing import Any, Iterable


def plot_predictions(
    model: Any,
    samples: Iterable[Any],
    *,
    n: int = 6,
    cols: int = 3,
    title: str = "Sample predictions",
):
    """Run `model.predict` on up to `n` samples and plot each with overlays.

    Returns the matplotlib Figure (so notebooks display it inline).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .eval.click import point_in_bbox

    samples = list(samples)[:n]
    if not samples:
        raise ValueError("no samples to visualize")
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.3))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, s in zip(axes, samples):
        out = model.predict(s.image, s.instruction)
        pt = out.parsed.point
        ok = bool(pt is not None and point_in_bbox(pt, s.bbox))
        ax.imshow(s.image)
        x1, y1, x2, y2 = s.bbox
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#16a34a", linewidth=2.2)
        )
        if pt is not None:
            ax.plot(
                pt[0], pt[1], marker="o", markersize=13,
                color="#16a34a" if ok else "#dc2626",
                markeredgecolor="white", markeredgewidth=1.6,
            )
        instr = (s.instruction or "")[:36]
        ax.set_title(("✓ " if ok else "✗ ") + instr, fontsize=9,
                     color="#15803d" if ok else "#b91c1c")
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
