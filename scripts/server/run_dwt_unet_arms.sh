#!/usr/bin/env bash
# Round 2 = Direction 1: the bijective wavelet U-Net (lossless Haar DWT down /
# IWT up replacing the lossy stride-2 down + nearest up between levels).
# Orthogonal to round 1 (a pure ARCHITECTURE change), so it can be queued without
# seeing the loss/equivariance numbers.  Same b64 anchor recipe, native-lr 0.
#
#   anchor  s1_b64               Y 34.1083 / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416
#   U1      --dwt-unet           wavelet down/up alone: does lossless stride-2 recover
#                                stride-1 fidelity (>=34.1083) at ~1/4 the activation?
#   U2      --dwt-unet +dwt-loss  stack the frequency loss on top (both wavelet levers).
#
# Computation-bound: NO --grad-ckpt.  batch 128 (wavelet down conv is 4c->c, only a
# few % more memory than the stride-2 conv it replaces).  +-0.05 dB is a tie; the
# real prize is matching stride-1 PSNR at stride-2 cost, so also read the wall-clock.
#
# Idempotent.  Launch (usually via the chained driver):  bash scripts/server/run_dwt_unet_arms.sh
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/wave_unet}"
mkdir -p "$OUT_ROOT" "$LOG"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-128}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

wait_gpu() {
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    echo "  waiting for GPU to drain: ${M} MiB"; sleep 10
  done
}

RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch $BATCH --amp --num-workers 12 --cache-data 0 \
  --lr 3e-4 --warmup 500 --steps $STEPS \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"

train_and_eval() {  # $1 name, rest = extra train flags
  local NAME="$1"; shift
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  (extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE "$@" > "$LOG/wu_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|unet native_lr|VAL |EARLY|Error|Traceback" "$LOG/wu_${NAME}.train.log" | tail -25
  local CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/wu_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/wu_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)
train_and_eval s1_b64_dwtunet      --dwt-unet
train_and_eval s1_b64_dwtunet_dw1  --dwt-unet --dwt-loss --dwt-weight 1 --dwt-levels 2
say "DWT-UNET ARMS DONE  total $(( ($(date +%s) - T0) / 60 )) min"
