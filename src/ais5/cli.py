"""CLI entrypoints exposed via `pyproject.toml [project.scripts]`.

    ais5-eval --config configs/eval/zero_shot_qwen.yaml
    ais5-train --config configs/train/lora_qwen.yaml
    ais5-bench --config configs/bench/efficiency.yaml
    ais5-papers --out ../Papers

Each command is a thin wrapper around the corresponding library function so
notebooks and CLIs share the exact same code path.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .utils import load_config, set_global_seed, setup_logging
from .utils.io import results_path, write_json
from .utils.logging import get_logger

app = typer.Typer(add_completion=False, help="ais5 — Small VLMs vs. GUI Specialists")
log = get_logger(__name__)


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    limit: int | None = typer.Option(None, help="Stop after N samples"),
    out: Path | None = typer.Option(None, help="Where to write results JSON"),
) -> None:
    """Run zero-shot click-accuracy eval defined by a YAML config."""
    setup_logging()
    cfg = load_config(config)
    set_global_seed(cfg.get("seed", 42))

    from .data import load_benchmark
    from .eval import evaluate_model
    from .models import get_model

    model_cfg = cfg["model"]
    bench_name = cfg["benchmark"]

    model = get_model(model_cfg["name"], **model_cfg.get("kwargs", {}))
    samples = load_benchmark(bench_name, **cfg.get("data", {}))
    run = evaluate_model(model, samples, benchmark=bench_name, limit=limit or cfg.get("limit"))

    payload = {
        "model": model.name,
        "benchmark": bench_name,
        "accuracy": run.accuracy,
        "n_samples": len(run.results),
        "avg_latency_ms": run.avg_latency_ms,
        "config": cfg,
    }
    target = Path(out) if out else results_path("eval", f"{bench_name}__{model.name}.json")
    write_json(payload, target)
    log.info("[bold green]accuracy=%.4f[/] (n=%d) → %s", run.accuracy, len(run.results), target)


@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    """LoRA fine-tune a generalist on grounding data."""
    setup_logging()
    cfg = load_config(config)
    set_global_seed(cfg.get("seed", 42))

    from .adapt import LoRAConfig, TrainingArgs, run_lora_training

    lora = LoRAConfig(**cfg["lora"])
    args = TrainingArgs(**cfg["training"])
    out_dir = run_lora_training(cfg["model"], lora, args)
    log.info("Adapter saved at %s", out_dir)


@app.command("bench")
def bench_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    limit: int | None = typer.Option(None),
) -> None:
    """Run the model × quantization × benchmark grid for Task 3."""
    setup_logging()
    cfg = load_config(config)
    set_global_seed(cfg.get("seed", 42))

    from .bench import run_full_benchmark
    from .data import load_benchmark
    from .models import get_model
    from .quant import resolve_quant_config

    rows = []
    for model_cfg in cfg["models"]:
        for quant_spec in cfg.get("quant", ["none"]):
            qc = resolve_quant_config(quant_spec)
            kwargs = dict(model_cfg.get("kwargs", {}))
            hf_quant = qc.to_hf()
            if hf_quant is not None:
                kwargs["quant_config"] = hf_quant
            model = get_model(model_cfg["name"], **kwargs)
            for bench in cfg["benchmarks"]:
                samples = list(load_benchmark(bench, **cfg.get("data", {})))
                result = run_full_benchmark(
                    model,
                    samples,
                    benchmark=bench,
                    quant_label=qc.name,
                    limit=limit or cfg.get("limit"),
                )
                rows.append(result.to_dict())
                log.info(
                    "%s | %s | %s → acc=%.4f mean_latency=%.1fms peak_vram=%.2fGB",
                    model.name,
                    qc.name,
                    bench,
                    result.accuracy,
                    result.latency.mean,
                    result.memory.peak_vram_gb,
                )

    target = results_path("bench", "summary.json")
    write_json({"runs": rows, "config": cfg}, target)
    log.info("Wrote %d benchmark rows to %s", len(rows), target)


@app.command("papers")
def papers_cmd(
    out: Path = typer.Option(Path("../Papers"), "--out", help="Output directory"),
) -> None:
    """(Re-)download the bibliography PDFs."""
    import runpy

    setup_logging()
    log.info("Delegating to scripts/download_papers.py")
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "download_papers.py"),
        run_name="__main__",
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
