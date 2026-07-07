"""Task 2 — resolution scaling + crop-then-click eval on ScreenSpot-Pro.

Wraps a Task-1 LoRA adapter so the eval runner's click-accuracy logic is reused
verbatim (no re-implementation of scoring in a notebook). For each sample:
  1. optionally rescale the screenshot by `scale` (0.5 / 1.0 / 2.0),
  2. run the model directly, or via the two-stage crop-then-click policy,
  3. map the predicted point back to ORIGINAL-image pixels before scoring.

    uv run python scripts/run_task2.py \
        --config configs/task2/scale2.0_crop512.yaml \
        --adapter checkpoints/qwen2.5-vl-3b-lora-r8 \
        --limit 20 --out results/task2/scale2.0_crop512.json
"""

from __future__ import annotations

from pathlib import Path

import typer

from ais5.utils import load_config, set_global_seed, setup_logging
from ais5.utils.io import write_json
from ais5.utils.logging import get_logger

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    adapter: Path | None = typer.Option(None, "--adapter", help="Task-1 LoRA adapter dir"),
    limit: int | None = typer.Option(None, "--limit"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    setup_logging()
    cfg = load_config(config)
    set_global_seed(cfg.get("seed", 42))

    from ais5.data import load_benchmark
    from ais5.eval import evaluate_model
    from ais5.models import get_model
    from ais5.models.base import GUIModel, ModelOutput
    from ais5.tile import CropConfig, crop_then_click, scale_image

    scale = float(cfg.get("scale", 1.0))
    crop_size = int(cfg.get("crop_size", 512))
    use_crop = bool(cfg.get("crop_then_click", True))
    bench = cfg["benchmark"]

    kwargs = {"device_map": "auto", "torch_dtype": "auto", "max_new_tokens": 64}
    if adapter is not None:
        kwargs["peft_adapter"] = str(adapter)
    base = get_model(cfg["model"], **kwargs)
    crop_cfg = CropConfig(crop_size=crop_size)

    class Task2Model(GUIModel):
        """Resolution-scaled, optionally crop-then-click wrapper over `base`.

        Predictions are returned in ORIGINAL-image coordinates so the shared
        runner scores them against the unscaled gold bbox.
        """

        name = f"{base.name}+scale{scale}+crop{crop_size if use_crop else 0}"
        family = base.family
        param_count_b = base.param_count_b

        def predict(self, image, instruction, **_kw):  # type: ignore[override]
            from dataclasses import replace

            work = scale_image(image, scale) if scale != 1.0 else image
            if use_crop:
                out_scaled = crop_then_click(base, work, instruction, cfg=crop_cfg)
            else:
                out_scaled = base.predict(work, instruction)
            pt = out_scaled.parsed.point
            if pt is not None and scale != 1.0:
                # Map the point on the scaled image back to original pixels.
                # ParsedAction is frozen, so build new instances rather than mutate.
                new_parsed = replace(out_scaled.parsed, point=(pt[0] / scale, pt[1] / scale))
                out_scaled = ModelOutput(
                    text=out_scaled.text, parsed=new_parsed, metadata=out_scaled.metadata
                )
            return out_scaled

    samples = load_benchmark(bench)
    run = evaluate_model(Task2Model(), samples, benchmark=bench, limit=limit or cfg.get("limit"))

    payload = {
        "model": Task2Model.name,
        "adapter": str(adapter) if adapter else None,
        "benchmark": bench,
        "scale": scale,
        "crop_size": crop_size if use_crop else None,
        "crop_then_click": use_crop,
        "accuracy": run.accuracy,
        "n_samples": len(run.results),
        "avg_latency_ms": run.avg_latency_ms,
        "config": cfg,
    }
    write_json(payload, out)
    log.info("task2 acc=%.4f (n=%d) scale=%s crop=%s -> %s",
             run.accuracy, len(run.results), scale, crop_size if use_crop else None, out)


if __name__ == "__main__":
    app()
