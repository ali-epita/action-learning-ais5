#!/usr/bin/env python3
"""Task 3 — efficiency x quant grid driver (headless, resumable).

Replaces the `ais5-bench` CLI loop for full-scale runs. The CLI bench command
loads a fresh model for every quant level WITHOUT freeing the previous one, and
`run_full_benchmark` reports `torch.cuda.max_memory_allocated`. So the bnb8/bnb4
cells' peak VRAM gets inflated by the still-resident fp16/bnb8 copies — and peak
VRAM is a headline Task 3 metric. This driver:

  * loads one (model, quant), runs all benchmarks for it, then `free_model()`
    before the next quant -> each cell's peak VRAM is measured in isolation;
  * passes measure_components=True so the vision-encode vs LLM-decode split is
    recorded (Task 3's required deeper-analysis component);
  * streams one row per (model, quant, benchmark) to a JSONL and skips cells
    already present, so a crash resumes where it stopped;
  * uses lazy iter_benchmark (one decoded image in RAM at a time).

    uv run python scripts/run_task3_grid.py -c configs/bench/efficiency_full.yaml
    uv run python scripts/run_task3_grid.py -c <cfg> --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ais5.bench import run_full_benchmark
from ais5.data import iter_benchmark
from ais5.models import get_model
from ais5.quant import resolve_quant_config
from ais5.utils import free_model, load_config, set_global_seed, setup_logging
from ais5.utils.logging import get_logger

log = get_logger(__name__)


def _done_keys(path: Path) -> set[tuple]:
    """(model, quant, benchmark) tuples already written to the JSONL."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Resume on the config registry name (model_key). Older rows only
            # carry the display name (e.g. "Qwen2.5-VL-3B-Instruct"), which
            # never equals a registry name, so pre-fix rows are re-run — the
            # same behavior as before this key existed, never worse.
            done.add((row.get("model_key") or row.get("model"), row.get("quant"), row.get("benchmark")))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 3 efficiency/quant grid")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--limit", type=int, default=None, help="override cfg limit (e.g. 20 for smoke)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(Path(args.config))
    set_global_seed(cfg.get("seed", 42))

    # Per-benchmark limits (e.g. full V2 but capped Pro/OSWorld). CLI --limit
    # overrides everything; else use cfg.limits[bench], else the global cfg.limit.
    cli_limit = args.limit
    global_limit = cfg.get("limit")
    per_bench_limits = cfg.get("limits", {}) or {}
    measure_components = cfg.get("measure_components", True)
    out_path = Path(cfg.get("out_jsonl", "results/bench/task3_grid.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    quants = cfg.get("quant", ["none"])
    benchmarks = cfg["benchmarks"]
    data_kwargs = cfg.get("data", {}) or {}

    done = _done_keys(out_path)
    log.info("Grid: %d models x %d quants x %d benchmarks; %d cells already done -> %s",
             len(models), len(quants), len(benchmarks), len(done), out_path)

    for model_cfg in models:
        mname = model_cfg["name"]
        for quant_spec in quants:
            qc = resolve_quant_config(quant_spec)
            if all((mname, qc.name, b) in done for b in benchmarks):
                log.info("[skip] %s | %s (all benchmarks done)", mname, qc.name)
                continue

            kwargs = dict(model_cfg.get("kwargs", {}))
            hf_quant = qc.to_hf()
            if hf_quant is not None:
                kwargs["quant_config"] = hf_quant

            try:
                model = get_model(mname, **kwargs)
            except Exception as e:  # noqa: BLE001 — a gated/missing model must not kill the grid
                log.error("[SKIP load] %s | %s -> %s", mname, qc.name, e)
                continue

            try:
                for b in benchmarks:
                    if (mname, qc.name, b) in done:
                        continue
                    blimit = cli_limit if cli_limit is not None else per_bench_limits.get(b, global_limit)
                    samples = iter_benchmark(b, limit=blimit, **data_kwargs)
                    result = run_full_benchmark(
                        model,
                        samples,
                        benchmark=b,
                        quant_label=qc.name,
                        limit=blimit,
                        measure_components=measure_components,
                    )
                    row = result.to_dict()
                    row["model_key"] = mname  # registry name: the resume key
                    with out_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                    done.add((mname, qc.name, b))
                    eff = {
                        k: round(v, 2)
                        for k, v in row.items()
                        if isinstance(v, (int, float))
                        and ("latency" in k or "vram" in k or "mem" in k)
                    }
                    log.info("%s | %s | %s -> acc=%.4f n=%d %s",
                             mname, qc.name, b, row.get("accuracy", 0.0),
                             row.get("n_samples", 0), eff)
            finally:
                # Free before the next quant/model so each cell's peak VRAM is clean.
                del model
                free_model()

    # Rebuild flat CSV/JSON from the JSONL (idempotent — safe to re-run).
    rows = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)
        log.info("Wrote %d rows -> %s (+ .csv/.json)", len(rows), out_path)


if __name__ == "__main__":
    main()
