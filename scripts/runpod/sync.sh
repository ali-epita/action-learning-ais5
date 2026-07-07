#!/usr/bin/env bash
# rsync helpers — run these FROM YOUR LAPTOP, not the pod.
#
# Configure the pod target once (RunPod gives you host+port for SSH):
#   export POD="root@<pod-ip>"            # e.g. root@149.36.x.x
#   export POD_PORT=22                    # RunPod often uses a high port
#   export POD_REPO=/workspace/action-learning-ais5   # on the persistent volume
#
# Then:
#   bash scripts/runpod/sync.sh up        # push repo to pod (excludes heavy dirs)
#   bash scripts/runpod/sync.sh down      # pull results + checkpoints + logs back
set -Eeuo pipefail
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
POD="${POD:?set POD=user@host}"
POD_PORT="${POD_PORT:-22}"
POD_REPO="${POD_REPO:-/workspace/action-learning-ais5}"
SSH="ssh -p $POD_PORT -o StrictHostKeyChecking=accept-new"

case "${1:-}" in
  up)
    # Push code + configs + .env. EXCLUDE: venv, git, downloaded data, results,
    # checkpoints, caches — those are big and/or pod-side only.
    rsync -avhz --progress -e "$SSH" \
      --exclude '.venv/' --exclude '.venv-ais5/' \
      --exclude '.git/' \
      --exclude '/results/' --exclude '/data/' --exclude '/checkpoints/' \
      --exclude '/logs/' --exclude '/wandb/' \
      --exclude '/.hf_cache/' \
      --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
      --exclude '.ipynb_checkpoints/' --exclude '.DS_Store' \
      "$LOCAL_REPO/" "$POD:$POD_REPO/"
    echo "pushed -> $POD:$POD_REPO"
    ;;
  down)
    # Pull just the artefacts worth keeping. JSON/markers/logs are tiny;
    # adapters are a few hundred MB each — include them, exclude HF cache.
    mkdir -p "$LOCAL_REPO/results" "$LOCAL_REPO/checkpoints" "$LOCAL_REPO/logs"
    rsync -avhz --progress -e "$SSH" \
      "$POD:$POD_REPO/results/" "$LOCAL_REPO/results/"
    rsync -avhz --progress -e "$SSH" \
      "$POD:$POD_REPO/logs/" "$LOCAL_REPO/logs/"
    rsync -avhz --progress -e "$SSH" \
      --include '*/' --include 'adapter_*' --include '*.json' --exclude '*' \
      "$POD:$POD_REPO/checkpoints/" "$LOCAL_REPO/checkpoints/"
    echo "pulled results/ logs/ checkpoints/(adapters) <- $POD:$POD_REPO"
    ;;
  *) echo "usage: $0 {up|down}"; exit 2 ;;
esac
