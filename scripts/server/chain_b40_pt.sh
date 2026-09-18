#!/usr/bin/env bash
# After the stride-1 sweep finishes, run b40 with BSRGAN pretraining.
#
# b40 (base 40, 7.58M params) is the strongest shape so far -- Y 34.007 on the
# official 100 pairs with NO pretraining, beating our modified SwinIR (33.47),
# the DiT (33.76) and stock SwinIR (32.97).  This chain answers the one question
# left: how much does pretraining add on top of that?
#
# Output goes to s1_pretrain/, NOT s1_sweep/, so the no-pretrain b40 result is
# not overwritten -- the comparison is the whole point.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_b40.log"
cd "$ROOT" || exit 1
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the stride-1 sweep to finish"
while pgrep -f "unet_s1_sweep" > /dev/null; do sleep 60; done
say "sweep finished"

# process exit does not hand VRAM back instantly; starting into a stale
# allocation OOMs on the first step and looks like a hang
for _ in $(seq 1 60); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  say "waiting for GPU to drain: ${M} MiB"
  sleep 10
done

say "starting b40 pretrain -> fine-tune"
OUT_ROOT="$ROOT/experiments/diffusion/s1_pretrain" \
SIZES="b40" \
PRETRAIN_SIZES="b40" \
PRE_STEPS="${PRE_STEPS:-15000}" \
  bash scripts/server/run_unet_s1_sweep.sh >> "$LOG" 2>&1
say "b40 pretrain chain done"
