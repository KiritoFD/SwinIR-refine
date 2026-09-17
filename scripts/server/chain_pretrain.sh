#!/usr/bin/env bash
# Wait for the current matrix (tmux realsr36h) to finish, then start the
# SwinIR/BSRGAN pretrain -> fine-tune run. Keeps the 4090 busy overnight
# instead of idling after arm 8.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_pretrain.log"
cd "$ROOT" || exit 1

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "chained: waiting for matrix session realsr36h to end"
for _ in $(seq 1 2880); do            # up to 24h
  tmux has-session -t realsr36h 2>/dev/null || break
  sleep 30
done
say "matrix session gone"

# pretraining data must be unpacked before we start
for _ in $(seq 1 240); do             # up to 2h for the download to land
  [[ -d "$ROOT/data/pretrain/DIV2K_train_HR" ]] && break
  sleep 30
done
if [[ ! -d "$ROOT/data/pretrain/DIV2K_train_HR" ]]; then
  say "ABORT: DIV2K_train_HR never appeared"
  exit 1
fi
n=$(find "$ROOT/data/pretrain" -type f \( -iname '*.png' -o -iname '*.jpg' \) | wc -l)
say "pretrain images available: $n"

tmux kill-session -t realsr36h 2>/dev/null
sleep 10
SESS=realsrpt RUN=run_pretrain.sh RUNLOG=pretrain.log bash scripts/server/start_36h.sh | tee -a "$LOG"
say "pretrain launched"
