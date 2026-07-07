#!/usr/bin/env bash
# =============================================================================
# AIS 5 — master orchestration
#
# Runs the WHOLE experimental pipeline in dependency order:
#
#   stage 0  baselines   4 zero-shot models on ScreenSpot-V2          (shared)
#   stage 1  task1       LoRA rank sweep {8,16,32,64} + adapter eval  (Task 1)
#   stage 2  task2       resolution / crop-then-click on Pro          (Task 2 ⇐ Task 1 adapters)
#   stage 3  task3       efficiency × quant grid                      (Task 3)
#
# Features
#   * --smoke         : run EVERY model/config at limit=20 first. Gate before full runs.
#   * per-job logfiles: $REPO_DIR/logs/<run_id>/<stage>__<job>.log
#   * resume          : a job that already produced its DONE marker is skipped.
#                       Re-run the script after a failure and it picks up where
#                       it stopped. Use --force to ignore markers.
#   * dependency gate : stage 2 refuses to start unless the r8 adapter from
#                       stage 1 exists.
#
# Designed to be launched inside tmux so it survives SSH disconnects — see
# scripts/runpod/tmux_launch.sh.
#
# Usage:
#   source /workspace/ais5.env
#   bash scripts/runpod/run_all.sh --smoke          # gate at limit=20
#   bash scripts/runpod/run_all.sh                  # full pipeline
#   bash scripts/runpod/run_all.sh --only task3     # one stage
#   bash scripts/runpod/run_all.sh --from task2     # resume from a stage
#   bash scripts/runpod/run_all.sh --force          # ignore DONE markers
# =============================================================================
set -Eeuo pipefail

# ── config ───────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
[[ -f "${VOL:-/workspace}/ais5.env" ]] && source "${VOL:-/workspace}/ais5.env"

SMOKE_LIMIT="${SMOKE_LIMIT:-20}"
RUN_ID="${AIS5_RUN_ID:-$(date +%Y-%m-%dT%H-%M-%S)}"
SMOKE=0; FORCE=0; ONLY=""; FROM=""
GENERALISTS=(qwen2.5-vl-3b paligemma-3b)
LORA_RANKS=(8 16 32 64)

# ── "Overnight" tier (defaults for a full, non-smoke run; ~12h on one A100) ──
# Right-sized for valid trends, not publication-grade. Override via env to scale.
#   TRAIN_SUBSET : LoRA training rows per run (50000 = full-scale)
#   TRAIN_BATCH/TRAIN_ACCUM : micro-batch x grad-accum (effective batch = product)
#   EVAL_SUBSET_HEAVY : cap for ScreenSpot-Pro / OSWorld-G evals (V2 stays full)
TRAIN_SUBSET="${TRAIN_SUBSET:-6000}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
TRAIN_ACCUM="${TRAIN_ACCUM:-8}"
EVAL_SUBSET_HEAVY="${EVAL_SUBSET_HEAVY:-500}"   # cap for ScreenSpot-Pro / OSWorld-G evals
EVAL_SUBSET_V2="${EVAL_SUBSET_V2:-}"            # empty = full V2 (1272); set to cap V2 evals (fast preview)
QUANTS="${QUANTS:-none,bnb8,bnb4}"              # Task-3 quant levels (fast preview may drop bnb8)

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --force) FORCE=1 ;;
    --only=*) ONLY="${arg#*=}" ;;
    --only)  shift; ONLY="${1:-}";;
    --from=*) FROM="${arg#*=}" ;;
    --from)  shift; FROM="${1:-}";;
    *) ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  LIMIT_FLAG=(--limit "$SMOKE_LIMIT"); TAG="smoke"; RUN_ID="smoke-$RUN_ID"
else
  LIMIT_FLAG=(); TAG="full"
fi

# V2-side eval cap (baselines + Task-1 adapter evals + the Task-3 V2 cell).
# smoke -> 20; else EVAL_SUBSET_V2 if set (fast preview); else full (1272).
if [[ "$SMOKE" == "1" ]]; then
  V2_FLAG=(--limit "$SMOKE_LIMIT"); V2_LIMIT_YAML="$SMOKE_LIMIT"
elif [[ -n "$EVAL_SUBSET_V2" ]]; then
  V2_FLAG=(--limit "$EVAL_SUBSET_V2"); V2_LIMIT_YAML="$EVAL_SUBSET_V2"
else
  V2_FLAG=(); V2_LIMIT_YAML="null"
fi
# YAML flow list for the Task-3 quant levels, e.g. [none, bnb8, bnb4].
QUANTS_YAML="[$(echo "$QUANTS" | sed 's/,/, /g')]"

LOG_DIR="$REPO_DIR/logs/$RUN_ID"
MARK_DIR="$REPO_DIR/results/$RUN_ID/.markers"
RES_DIR="$REPO_DIR/results/$RUN_ID"
CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$RES_DIR"

echo "============================================================"
echo " AIS5 pipeline  run_id=$RUN_ID  mode=$TAG  limit=${SMOKE_LIMIT:-none}"
echo " logs    -> $LOG_DIR"
echo " results -> $RES_DIR"
echo "============================================================"

# ── helpers ──────────────────────────────────────────────────────────────────
# run_job <stage> <name> -- <command...>
# Logs to its own file, writes a DONE marker on success, skips if already done.
run_job() {
  local stage="$1" name="$2"; shift 2
  [[ "$1" == "--" ]] && shift
  local key="${stage}__${name}"
  local marker="$MARK_DIR/$key.done"
  local log="$LOG_DIR/$key.log"

  if [[ -f "$marker" && "$FORCE" != "1" ]]; then
    echo "  [skip] $key (marker present)"
    return 0
  fi

  echo "  [run ] $key  -> $log"
  local start; start=$(date +%s)
  # Stream to the per-job log AND keep stdout for the tmux pane.
  if "$@" >>"$log" 2>&1; then
    local dur=$(( $(date +%s) - start ))
    echo "ok $(date -u +%FT%TZ) ${dur}s :: $*" > "$marker"
    echo "  [ok  ] $key (${dur}s)"
  else
    local rc=$?
    echo "  [FAIL] $key rc=$rc — see $log"
    # baselines + task3 are per-model and non-fatal: one model failing to load
    # (e.g. an arch the wrapper doesn't yet support) must not sink the other
    # models or the downstream stages. No .done marker is written, so a later
    # re-run retries the cell once it's fixed.
    if [[ "$stage" == "baselines" || "$stage" == "task3" ]]; then
      echo "  [warn] $stage non-fatal — skipping this cell, continuing pipeline."
      echo "fail rc=$rc $(date -u +%FT%TZ) :: $*" > "$MARK_DIR/$key.failed"
      return 0
    fi
    echo "         pipeline halts; fix and re-run (completed jobs are skipped)."
    return $rc
  fi
}

# --no-sync: use the bootstrapped venv as-is. A plain `uv run` re-syncs to the
# lockfile and would prune the pip-installed bitsandbytes/wandb (not in pyproject).
uvrun() { uv run --no-sync python -m ais5.cli "$@"; }

stage_enabled() {
  local s="$1"
  [[ -n "$ONLY" && "$ONLY" != "$s" ]] && return 1
  if [[ -n "$FROM" ]]; then
    local order=(baselines task1 task2 task3)
    local from_i=-1 this_i=-1 i=0
    for x in "${order[@]}"; do
      [[ "$x" == "$FROM" ]] && from_i=$i
      [[ "$x" == "$s" ]] && this_i=$i
      i=$((i+1))
    done
    [[ "$this_i" -lt "$from_i" ]] && return 1
  fi
  return 0
}

# =============================================================================
# STAGE 0 — shared zero-shot baselines (4 models on ScreenSpot-V2)
# Run each model in its own process so VRAM never accumulates.
# =============================================================================
if stage_enabled baselines; then
  echo "── stage 0: baselines ───────────────────────────────────────"
  run_job baselines qwen2.5-vl-3b -- \
    uvrun eval -c configs/eval/zero_shot_qwen.yaml "${V2_FLAG[@]}" \
      --out "$RES_DIR/baselines/screenspot-v2__qwen2.5-vl-3b.json"

  run_job baselines paligemma-3b -- \
    uvrun eval -c configs/eval/zero_shot_paligemma.yaml "${V2_FLAG[@]}" \
      --out "$RES_DIR/baselines/screenspot-v2__paligemma-3b.json"

  run_job baselines os-atlas-4b -- \
    uvrun eval -c configs/eval/specialists_screenspot.yaml "${V2_FLAG[@]}" \
      --out "$RES_DIR/baselines/screenspot-v2__os-atlas-4b.json"

  run_job baselines showui-2b -- \
    uvrun eval -c configs/eval/showui_screenspot.yaml "${V2_FLAG[@]}" \
      --out "$RES_DIR/baselines/screenspot-v2__showui-2b.json"
fi

# =============================================================================
# STAGE 1 — Task 1: LoRA rank sweep + adapter eval
# For each generalist × rank: train an adapter, then eval it on V2 (+Pro full).
# Training writes checkpoints/<model>-lora-r<k>/; the adapter eval points the
# model wrapper at that dir via kwargs.peft_adapter (passed through a per-job
# config generated on the fly).
# In --smoke mode we set a tiny train_subset so each cell finishes in minutes.
# =============================================================================
if stage_enabled task1; then
  echo "── stage 1: task1 LoRA rank sweep ───────────────────────────"
  for model in "${GENERALISTS[@]}"; do
    for r in "${LORA_RANKS[@]}"; do
      out_dir="$CKPT_DIR/${model}-lora-r${r}"
      train_cfg="$RES_DIR/task1/cfg/train_${model}_r${r}.yaml"
      eval_cfg="$RES_DIR/task1/cfg/eval_${model}_r${r}.yaml"
      mkdir -p "$(dirname "$train_cfg")"

      # subset: TRAIN_SUBSET full, 200 smoke (enough steps to write a valid adapter).
      subset=$([[ "$SMOKE" == "1" ]] && echo 200 || echo "$TRAIN_SUBSET")
      batch=$([[ "$SMOKE" == "1" ]] && echo 1 || echo "$TRAIN_BATCH")
      accum=$([[ "$SMOKE" == "1" ]] && echo 16 || echo "$TRAIN_ACCUM")
      alpha=$(( r * 2 ))

      cat > "$train_cfg" <<YAML
seed: 42
model: ${model}
lora: {r: ${r}, alpha: ${alpha}, dropout: 0.05, bias: none, backbone: ${model%%-3b}}
training:
  output_dir: ${out_dir}
  train_dataset: osunlp/UGround-V1-Data
  adapter: uground
  train_subset_size: ${subset}
  num_train_epochs: 1.0
  per_device_train_batch_size: ${batch}
  gradient_accumulation_steps: ${accum}
  learning_rate: 5.0e-5
  warmup_ratio: 0.03
  bf16: true
  fp16: false
  logging_steps: 25
  save_steps: 1000
  report_to: [wandb]
  run_name: ${model}-r${r}-${TAG}
YAML

      cat > "$eval_cfg" <<YAML
seed: 42
benchmark: screenspot-v2
model:
  name: ${model}
  kwargs: {device_map: auto, torch_dtype: auto, max_new_tokens: 64, peft_adapter: ${out_dir}}
data: {}
YAML

      # Train (skipped on resume if adapter file already exists).
      if [[ -f "$out_dir/adapter_model.safetensors" && "$FORCE" != "1" ]]; then
        echo "  [skip] task1__train_${model}_r${r} (adapter present)"
        echo "ok preexisting" > "$MARK_DIR/task1__train_${model}_r${r}.done"
      else
        run_job task1 "train_${model}_r${r}" -- uvrun train -c "$train_cfg"
      fi

      # Eval the adapter on V2 (smoke: 20; fast: EVAL_SUBSET_V2; else full set).
      run_job task1 "eval_${model}_r${r}_v2" -- \
        uvrun eval -c "$eval_cfg" "${V2_FLAG[@]}" \
          --out "$RES_DIR/task1/screenspot-v2__${model}-lora-r${r}.json"
    done
  done
fi

# =============================================================================
# STAGE 2 — Task 2: resolution scaling + crop-then-click (DEPENDS ON Task 1)
# Uses the best generalist adapter from stage 1 (default r8 Qwen). Sweeps
# scale {0.5,1,2} × crop {512,768} on ScreenSpot-Pro. Driven by a small CLI
# subcommand-free runner script that imports ais5.tile.
# =============================================================================
if stage_enabled task2; then
  echo "── stage 2: task2 resolution/crop (Pro) ─────────────────────"
  BASE_ADAPTER="$CKPT_DIR/qwen2.5-vl-3b-lora-r8"
  if [[ ! -f "$BASE_ADAPTER/adapter_model.safetensors" ]]; then
    echo "  [FAIL] task2 dependency missing: $BASE_ADAPTER/adapter_model.safetensors"
    echo "         run stage 1 (task1) first, or --only task1."
    exit 3
  fi
  # ScreenSpot-Pro is huge + crop doubles inference; cap to the heavy-eval subset.
  T2_LIMIT=$([[ "$SMOKE" == "1" ]] && echo "$SMOKE_LIMIT" || echo "$EVAL_SUBSET_HEAVY")
  for cfg in configs/task2/*.yaml; do
    [[ -e "$cfg" ]] || continue
    name="$(basename "$cfg" .yaml)"
    run_job task2 "$name" -- \
      uv run --no-sync python scripts/run_task2.py --config "$cfg" \
        --adapter "$BASE_ADAPTER" --limit "$T2_LIMIT" \
        --out "$RES_DIR/task2/${name}.json"
  done
fi

# =============================================================================
# STAGE 3 — Task 3: efficiency × quant grid (the owner's task)
# 4 models × {none,bnb8,bnb4} × {V2,Pro,OSWorld-G}. The bench command loads one
# model at a time and records accuracy + latency + peak VRAM. We split the grid
# by model so a crash in one model doesn't lose the others, and each model gets
# its own JSON + marker (= per-model resume).
# =============================================================================
if stage_enabled task3; then
  echo "── stage 3: task3 efficiency/quant grid ─────────────────────"
  for model in qwen2.5-vl-3b paligemma-3b showui-2b os-atlas-4b; do
    cfg="$RES_DIR/task3/cfg/${model}.yaml"
    mkdir -p "$(dirname "$cfg")"
    # Smoke stays on V2 only at limit; full run adds Pro + OSWorld-G.
    if [[ "$SMOKE" == "1" ]]; then
      cat > "$cfg" <<YAML
seed: 42
limit: ${SMOKE_LIMIT}
out_jsonl: ${RES_DIR}/task3/task3_grid.jsonl
measure_components: true
models:
  - name: ${model}
    kwargs: {device_map: auto, torch_dtype: auto}
quant: [none, bnb8, bnb4]
benchmarks:
  - screenspot-v2
data: {}
YAML
    else
      cat > "$cfg" <<YAML
seed: 42
limit: null
limits:
  screenspot-v2: ${V2_LIMIT_YAML}
  screenspot-pro: ${EVAL_SUBSET_HEAVY}
  osworld-g: ${EVAL_SUBSET_HEAVY}
out_jsonl: ${RES_DIR}/task3/task3_grid.jsonl
measure_components: true
models:
  - name: ${model}
    kwargs: {device_map: auto, torch_dtype: auto}
quant: ${QUANTS_YAML}
benchmarks:
  - screenspot-v2
  - screenspot-pro
  - osworld-g
data: {}
YAML
    fi
    # Use the dedicated grid driver (NOT `ais5-bench`): it frees the model
    # between every quant level so each cell's peak VRAM is measured clean,
    # records the encoder/decoder split, and resumes per-(model,quant,bench).
    # All 4 models append to the one $RES_DIR/task3/task3_grid.jsonl.
    run_job task3 "$model" -- \
      uv run --no-sync python scripts/run_task3_grid.py -c "$cfg" "${LIMIT_FLAG[@]}"
  done
fi

echo "============================================================"
echo " DONE  ($TAG)  run_id=$RUN_ID"
echo " markers: $MARK_DIR"
if [[ "$SMOKE" == "1" ]]; then
  echo
  echo " Smoke gate passed. Review the per-job logs, then run the full set:"
  echo "   bash scripts/runpod/run_all.sh"
fi
echo "============================================================"
