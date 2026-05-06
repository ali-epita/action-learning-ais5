#!/usr/bin/env python3
"""End-to-end smoke demo of the ais5 library.

No GPU, no network, no dataset download. Used for the modeling-checkpoint
meeting to show that the shared infrastructure (model registry, prompt parser,
eval harness, Pareto fronts) is wired correctly. The actual model inference
and LoRA training runs on Colab/Kaggle — see notebooks/.

    uv run python scripts/demo.py
"""

from __future__ import annotations

from PIL import Image

from ais5 import __version__
from ais5.bench import ParetoPoint, pareto_front
from ais5.data.types import GroundingSample
from ais5.eval import by_ui_type, evaluate_model
from ais5.models import list_models
from ais5.models.base import GUIModel, ModelOutput
from ais5.prompt.action import ParsedAction, parse_click


def header(text: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {text}")
    print("─" * 72)


# ── 1. Model registry ─────────────────────────────────────────────────────────


def section_models() -> None:
    header("Registered models — single registry, lazy heavy imports")
    print(f"  {'name':<24s} {'family':<11s} {'params':>7s}   HF id")
    print(f"  {'-' * 24} {'-' * 11} {'-' * 7}   {'-' * 36}")
    for m in list_models():
        print(f"  {m.name:<24s} {m.family:<11s} {m.param_count_b:5.1f}B   {m.hf_id}")


# ── 2. Unified action parser ──────────────────────────────────────────────────


def section_parser() -> None:
    header("parse_click — every model dialect goes through one parser")
    cases: list[tuple[str, str, tuple[int, int] | None]] = [
        ("Qwen2.5-VL",        "<click>423, 167</click>",                            None),
        ("OS-Atlas",          "The save icon is located at (256, 512).",            None),
        ("ShowUI (JSON)",     '{"action": "click", "x": 80, "y": 240}',             None),
        ("PaliGemma",         "<loc0512><loc0768>",                                 (1024, 1024)),
        ("Box centroid",      "<box>100, 200, 300, 400</box>",                      None),
        ("Unparseable",       "no idea where to click",                             None),
    ]
    print(f"  {'dialect':<16s} {'parser':<16s} point")
    print(f"  {'-' * 16} {'-' * 16} {'-' * 30}")
    for name, text, size in cases:
        p = parse_click(text, image_size=size)
        point = f"{p.point}" if p.point else "—"
        print(f"  {name:<16s} {p.parser:<16s} {point}")


# ── 3. End-to-end eval with a mock model ──────────────────────────────────────


class MockModel(GUIModel):
    """Deterministic stand-in: clicks (50, 50) on odd calls, far away on even."""

    name = "mock-3b"
    param_count_b = 3.0
    family = "generalist"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: Image.Image, instruction: str, **_: object) -> ModelOutput:
        self.calls += 1
        # (50, 50) lies inside (0, 0, 100, 100); (999, 999) does not.
        x, y = (50.0, 50.0) if self.calls % 2 else (999.0, 999.0)
        return ModelOutput(
            text=f"<click>{x}, {y}</click>",
            parsed=ParsedAction(point=(x, y), raw="", parser="mock"),
        )


def section_eval() -> None:
    header("eval harness — model + benchmark → click_accuracy + breakdowns")
    img = Image.new("RGB", (1000, 1000), color="white")
    samples = [
        GroundingSample(
            image=img,
            instruction=f"click element {i}",
            bbox=(0, 0, 100, 100),
            image_size=(1000, 1000),
            benchmark="screenspot-v2",
            target_type="icon" if i % 2 else "text",
            ui_type=["web", "desktop", "mobile"][i % 3],
            sample_id=str(i),
        )
        for i in range(9)
    ]
    run = evaluate_model(MockModel(), samples, benchmark="screenspot-v2", progress=False)

    print(f"  accuracy        = {run.accuracy:.3f}  (n={len(run.results)})")
    print(f"  avg latency     = {run.avg_latency_ms:.2f} ms  (mock; real models later)")
    print()
    print("  breakdown by UI type:")
    for _, row in by_ui_type(run.results).iterrows():
        print(
            f"    {row['ui']:<8s} {int(row['correct'])}/{int(row['total'])}"
            f"   acc={row['accuracy']:.3f}"
        )


# ── 4. Pareto front ───────────────────────────────────────────────────────────


def section_pareto() -> None:
    header("Pareto front — illustrative latency vs. accuracy (Task 3 output)")
    # Numbers chosen for readability, not real measurements.
    pts = [
        ParetoPoint(label="Qwen2.5-VL-3B fp16",   accuracy=0.41, cost=1850),
        ParetoPoint(label="Qwen2.5-VL-3B int8",   accuracy=0.40, cost=1420),
        ParetoPoint(label="Qwen2.5-VL-3B nf4",    accuracy=0.37, cost=1100),
        ParetoPoint(label="PaliGemma-3B fp16",    accuracy=0.35, cost=1300),
        ParetoPoint(label="ShowUI-2B fp16",       accuracy=0.84, cost=1620),
        ParetoPoint(label="OS-Atlas-4B fp16",     accuracy=0.87, cost=2200),
    ]
    front_labels = {p.label for p in pareto_front(pts)}

    print(f"  {'config':<24s}  {'accuracy':>9s}  {'cost (ms)':>10s}   pareto?")
    print(f"  {'-' * 24}  {'-' * 9}  {'-' * 10}   {'-' * 7}")
    for p in pts:
        marker = "★" if p.label in front_labels else " "
        print(f"  {p.label:<24s}  {p.accuracy:9.3f}  {p.cost:10.0f}      {marker}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n  ais5 v{__version__} — Small VLMs vs. GUI Specialists\n")
    section_models()
    section_parser()
    section_eval()
    section_pareto()
    print(f"\n{'─' * 72}")
    print("  Next on Colab: notebooks/00_baseline_qwen_screenspot_v2.ipynb")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    main()
