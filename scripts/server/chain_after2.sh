#!/usr/bin/env bash
# Everything queued behind the main matrix (tmux realsr36h):
#   1. latent_flow_res  -- closes the residual confound in the headline result
#   2. BSRGAN pretrain -> RealSR fine-tune -> official eval
#
# Replaces chain_after.sh.  Three differences over v1:
#   * kill a stale SESS before launching.  start_36h.sh refuses to create a
#     session that already exists, so the dead 'realsrpt' left behind by the
#     killed chain_pretrain.sh would have silently swallowed stage 2.
#   * launch stage B directly instead of via start_36h.sh, so STEPS1 really
#     reaches run_pretrain.sh inside the tmux session.
#   * size stage 1 from the wall-clock time still available, so a late matrix
#     cannot push the whole experiment past its window.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_after.log"
cd "$ROOT" || exit 1

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

wait_gone() {                        # $1 = tmux session name
  local i
  for i in $(seq 1 2880); do         # up to 24h
    tmux has-session -t "$1" 2>/dev/null || return 0
    sleep 30
  done
  return 1
}

say "chained(v2): waiting for matrix session realsr36h to end"
wait_gone realsr36h
tmux kill-session -t realsr36h 2>/dev/null
sleep 10
say "matrix session gone"

# ---------------------------------------------------------- stage A
say "--- stage A: latent_flow_res ---"
tmux kill-session -t realsr_extra 2>/dev/null
SESS=realsr_extra RUN=run_extra_arms.sh RUNLOG=extra_arms.log \
  bash scripts/server/start_36h.sh | tee -a "$LOG"
wait_gone realsr_extra
tmux kill-session -t realsr_extra 2>/dev/null
say "extra arms done"

# ---------------------------------------------------------- stage B
say "--- stage B: BSRGAN pretrain -> fine-tune ---"
for _ in $(seq 1 240); do
  [[ -d "$ROOT/data/pretrain/DIV2K_train_HR" ]] && break
  sleep 30
done

# Size stage 1 from the time left before the horizon, reserving RESERVE_S for
# the fine-tune + official eval.
# An absolute wall clock, not "N hours from now" -- a relative horizon computed
# at stage-B start is useless, because by then most of the window is already
# gone and it would happily grant another full day.
#
# WALL must be a format `date -d` actually accepts.  A bare "MM-DD HH:MM" is
# NOT one (GNU date: "invalid date") -- that mistake silently dropped STEPS1 to
# the MINSTEPS floor on the first run of this script.  Use YYYY-MM-DD HH:MM.
WALL="${WALL:-2026-09-18 11:00}"     # 36h after the original 09-16 23:07 launch
RESERVE_S="${RESERVE_S:-13000}"      # ~3.6h: 16k fine-tune @0.75s + official eval
S_PER_STEP="${S_PER_STEP:-0.90}"     # compiled pixel-S b=28, BSRGAN data on the fly
MAXSTEPS="${STEPS1:-30000}"
MINSTEPS="${MINSTEPS1:-8000}"
FALLBACK_H="${FALLBACK_H:-10}"

now=$(date +%s)
wall=$(date -d "$WALL" +%s 2>/dev/null)
if [[ -z "${wall:-}" ]]; then
  wall=$(( now + FALLBACK_H * 3600 ))
  say "WARNING: cannot parse WALL='$WALL' -- falling back to ${FALLBACK_H}h from now"
fi
avail=$(( wall - now ))
budget=$(( avail - RESERVE_S ))
steps=$(awk -v b="$budget" -v s="$S_PER_STEP" \
        'BEGIN{printf "%d", (b>0 ? b/s : 0)}')
(( steps > MAXSTEPS )) && steps=$MAXSTEPS
(( steps < MINSTEPS )) && steps=$MINSTEPS
say "budget: wall=$WALL ($(date -d @$wall +'%m-%d %H:%M')) avail=${avail}s reserve=${RESERVE_S}s -> STEPS1=$steps (max $MAXSTEPS)"
if (( steps <= MINSTEPS )); then
  say "WARNING: STEPS1 hit the ${MINSTEPS}-step floor -- the window is nearly gone,"
  say "         so this pretrain will be too short to be worth much.  Check WALL."
fi

tmux kill-session -t realsrpt 2>/dev/null
tmux new-session -d -s realsrpt \
  "cd $ROOT && export STEPS1=$steps STEPS2=${STEPS2:-16000} && \
   bash scripts/server/run_pretrain.sh 2>&1 | tee experiments/diffusion/pretrain.log"
sleep 3
if tmux has-session -t realsrpt 2>/dev/null; then
  say "pretrain launched in tmux 'realsrpt' (STEPS1=$steps STEPS2=${STEPS2:-16000})"
else
  say "ERROR: pretrain session did not start"
fi

wait_gone realsrpt
say "pretrain session finished"
