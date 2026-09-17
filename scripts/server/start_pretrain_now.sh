#!/usr/bin/env bash
# Relaunch stage B (BSRGAN pretrain -> RealSR fine-tune -> official eval) with a
# step count sized from an absolute wall clock.
#
# Split out of chain_after2.sh because that script's first run got STEPS1 wrong:
# `date -d "09-18 11:00"` is an invalid date for GNU date, the `|| echo 0`
# fallback swallowed it, and the "wall already passed" branch then handed the
# pretrain the 8000-step floor.  A silent clamp that *looks* successful is the
# worst possible failure mode, so this script parses strictly and refuses to
# guess: no wall, no launch.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_after.log"
cd "$ROOT" || exit 1

WALL="${WALL:-2026-09-18 11:00}"     # must be YYYY-MM-DD HH:MM
RESERVE_S="${RESERVE_S:-13000}"      # ~3.6h: 16k fine-tune @0.75s + official eval
S_PER_STEP="${S_PER_STEP:-0.90}"     # compiled pixel-S b=28, BSRGAN data on the fly
MAXSTEPS="${STEPS1:-30000}"
MINSTEPS="${MINSTEPS1:-8000}"

now=$(date +%s)
wall=$(date -d "$WALL" +%s 2>/dev/null)
if [[ -z "${wall:-}" ]]; then
  echo "FATAL: cannot parse WALL='$WALL' (need YYYY-MM-DD HH:MM). Refusing to guess."
  exit 1
fi
avail=$(( wall - now ))
budget=$(( avail - RESERVE_S ))
steps=$(awk -v b="$budget" -v s="$S_PER_STEP" 'BEGIN{printf "%d", (b>0 ? b/s : 0)}')
(( steps > MAXSTEPS )) && steps=$MAXSTEPS
(( steps < MINSTEPS )) && steps=$MINSTEPS

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
say "stage B relaunch: wall=$WALL ($(date -d @$wall +'%m-%d %H:%M')) avail=${avail}s reserve=${RESERVE_S}s -> STEPS1=$steps"
if (( steps <= MINSTEPS )); then
  say "WARNING: STEPS1 hit the ${MINSTEPS} floor; the window is nearly gone."
fi

tmux kill-session -t realsrpt 2>/dev/null
tmux new-session -d -s realsrpt \
  "cd $ROOT && export STEPS1=$steps STEPS2=${STEPS2:-16000} && \
   bash scripts/server/run_pretrain.sh 2>&1 | tee experiments/diffusion/pretrain.log"
sleep 3
if tmux has-session -t realsrpt 2>/dev/null; then
  say "launched tmux 'realsrpt' with STEPS1=$steps STEPS2=${STEPS2:-16000}"
else
  say "ERROR: realsrpt session did not start"
  exit 1
fi
