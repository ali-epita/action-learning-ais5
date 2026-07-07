#!/usr/bin/env bash
# Launch the pipeline inside a detached tmux session so it survives SSH drops.
#   bash scripts/runpod/tmux_launch.sh --smoke    # gate
#   bash scripts/runpod/tmux_launch.sh            # full run
# Re-attach later from any SSH session:  tmux attach -t ais5
# Tail without attaching:                tail -f $REPO_DIR/logs/<run_id>/*.log
set -Eeuo pipefail
VOL="${VOL:-/workspace}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION="ais5"

command -v tmux >/dev/null || { apt-get update && apt-get install -y tmux; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists — attaching."
  exec tmux attach -t "$SESSION"
fi

# Pane runs: source env, then the master script with whatever flags were passed.
tmux new-session -d -s "$SESSION" -c "$REPO_DIR" \
  "source $VOL/ais5.env; bash scripts/runpod/run_all.sh $* 2>&1 | tee $REPO_DIR/logs/run_all.console.log; echo; echo '=== pipeline exited (rc=$?) — press enter to close ==='; read"
echo "started tmux session '$SESSION'. attach with:  tmux attach -t $SESSION"
