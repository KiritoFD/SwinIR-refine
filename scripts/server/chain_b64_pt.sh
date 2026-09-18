#!/usr/bin/env bash
# b64 (the strongest shape, Y 34.108 with no pretraining) with BSRGAN
# pretraining on top, then the RealSR fine-tune and the official eval.
#
# Fixes over the b40 attempt:
#   * DIV2K_valid is held out of the pretrain train set and used as the
#     PRETRAIN val, so ckpt_best is picked on pretraining progress instead of
#     on RealSR domain transfer (which is what it was doing before, and is why
#     the b40 pretrain "peaked" at step 7000 and then drifted).
#   * pretrain batch is per-size (128 for b64), so it no longer OOMs.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
LOG="$ROOT/experiments/chain_b64.log"
cd "$ROOT" || exit 1
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# Files written from the Windows side carry CRLF and bash then reads `cd "$ROOT"`
# as `cd $'/home/ds/realsr\r'` and dies.  Normalise every script here rather
# than relying on remembering to sed after each scp.
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} +
say "normalised CRLF in scripts/server/*.sh"

for _ in $(seq 1 60); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  say "waiting for GPU to drain: ${M} MiB"
  sleep 10
done

say "starting b64 pretrain -> fine-tune -> eval"
OUT_ROOT="$ROOT/experiments/diffusion/s1_pretrain_b64" \
SIZES="b64" \
PRETRAIN_SIZES="b64" \
PRE_STEPS="${PRE_STEPS:-25000}" \
  bash scripts/server/run_unet_s1_sweep.sh >> "$LOG" 2>&1
say "b64 pretrain chain done"
