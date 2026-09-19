#!/usr/bin/env bash
# Mamba A/B: stride-1 VSS (2D selective scan) vs the s1_b64 anchor.
#
#   anchor  s1_b64  U-Net 18.67M, zero pretrain:  Y 34.1083 / SSIM 0.9246
#           (with BSRGAN pretrain the line tops out at 34.1988)
#
#   M1 mamba_c128   dim 128, 4 RG x 4 VSS, d_state 16, expand 2  (2.75M)
#                   the user-spec topology; NOTE the real param count is
#                   2.75M, not 25-30M (Mamba blocks are projection-lean)
#   M2 mamba_c256   dim 256, same topology                        (10.48M)
#                   capacity-matched to the "inflection >10M" finding
#
# Sample budget matches the anchor's exactly (1.28M = batch x steps):
#   CUDA kernel available (mamba_ssm):  c128 batch 128 x 10000 (anchor-exact)
#   pure-torch scan fallback:           c128/c256 batch 64 x 20000
# Everything else verbatim from the anchor recipe (lr 3e-4, warmup 500, EMA
# 0.999, amp, val 16 pairs, patience 15).
#
# Runs AFTER the new-arms matrix (waits for the 'newarms' tmux session).
# Idempotent per arm via eval_iqa/eval.json.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/mamba}"
mkdir -p "$OUT_ROOT" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

wait_gpu() {
  for _ in $(seq 1 120); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    echo "  waiting for GPU to drain: ${M} MiB"; sleep 15
  done
}

if "$PY" -c "import mamba_ssm" 2>/dev/null; then
  BACKEND="mamba_ssm"; C128_BATCH=128; C128_STEPS=10000
else
  BACKEND="torch";     C128_BATCH=64;  C128_STEPS=20000
fi
say "mamba backend=$BACKEND  c128 protocol: batch $C128_BATCH x $C128_STEPS"

RECIPE="--backbone mamba --objective reg --residual 1 \
  --decoded-manifest data/decoded/manifest.json --ssm-backend $BACKEND \
  --num-groups 4 --num-res 4 --ssm-state 16 --ssm-expand 2 \
  --lr-patch 64 --amp --num-workers 12 --cache-data 0 --grad-ckpt \
  --lr 3e-4 --warmup 500 \
  --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"

train_and_eval() {  # $1 name, $2 batch, $3 steps, rest = extra flags
  local NAME="$1" BATCH="$2" STEPS="$3"; shift 3
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  batch=$BATCH steps=$STEPS extra=$*"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE --base "$DIM" --batch "$BATCH" --steps "$STEPS" "$@" \
    > "$LOG/mb_${NAME}.train.log" 2>&1
  grep -E "MambaSR|mamba dim|VAL |EARLY|Error|Traceback" "$LOG/mb_${NAME}.train.log" | tail -25
  local CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/mb_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mb_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)

DIM=128
train_and_eval mamba_c128 "$C128_BATCH" "$C128_STEPS"

if [ "${RUN_C256:-0}" = "1" ] || [ "$BACKEND" = "mamba_ssm" ]; then
  DIM=256
  train_and_eval mamba_c256 64 20000
else
  say "SKIP mamba_c256 -- pure-torch scan is ~5x slower; set RUN_C256=1 to force"
fi

say "MAMBA ARMS DONE  total $(( ($(date +%s) - T0) / 60 )) min"
