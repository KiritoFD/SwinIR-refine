#!/usr/bin/env bash
# Everything queued behind the main matrix (tmux realsr36h):
#   1. latent_flow_res  -- closes the residual confound in the headline result
#   2. BSRGAN pretrain -> RealSR fine-tune -> official eval
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_after.log"
cd "$ROOT" || exit 1

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "chained: waiting for matrix session realsr36h to end"
for _ in $(seq 1 2880); do            # up to 24h
  tmux has-session -t realsr36h 2>/dev/null || break
  sleep 30
done
say "matrix session gone"
tmux kill-session -t realsr36h 2>/dev/null
sleep 10

say "--- stage A: latent_flow_res ---"
SESS=realsr_extra RUN=run_extra_arms.sh RUNLOG=extra_arms.log bash scripts/server/start_36h.sh | tee -a "$LOG"

for _ in $(seq 1 2880); do
  tmux has-session -t realsr_extra 2>/dev/null || break
  sleep 30
done
tmux kill-session -t realsr_extra 2>/dev/null
say "extra arms done"

say "--- stage B: BSRGAN pretrain -> fine-tune ---"
for _ in $(seq 1 240); do             # give the Flickr2K extract time to land
  [[ -d "$ROOT/data/pretrain/DIV2K_train_HR" ]] && break
  sleep 30
done
SESS=realsrpt RUN=run_pretrain.sh RUNLOG=pretrain.log bash scripts/server/start_36h.sh | tee -a "$LOG"
say "pretrain launched"
