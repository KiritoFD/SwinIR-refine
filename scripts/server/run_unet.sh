#!/usr/bin/env bash
# One U-Net run on RealSR (no pretraining), then the official 100-pair eval.
#
# The shape is driven by env vars so the same script covers the whole sweep:
#   TAG=unet_1128 BASE=64 MULT=1,1,2,8 NUM_RES=2 STEPS=10000 \
#     BATCH=390 bash scripts/server/run_unet.sh
#
# Reference shape (mult 1,2,4,4, base 64) -> 18.68M params, 11.41 GFLOP/sample.
# The shapes under test keep the parameter count but move capacity to lower
# resolution, where a weight costs 64x fewer FLOPs (level 3 is 8x8, level 0 is
# 64x64).  See scripts/server/bench_unet_shapes.py.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

TAG="${TAG:-unet_reg}"
OUT="$ROOT/experiments/diffusion/$TAG"
LOG="$ROOT/experiments/diffusion/logs"
mkdir -p "$OUT" "$LOG"

SHAPE_ARGS=""
[[ -n "${BASE:-}" ]] && SHAPE_ARGS="$SHAPE_ARGS --base $BASE"
[[ -n "${MULT:-}" ]] && SHAPE_ARGS="$SHAPE_ARGS --mult $MULT"
[[ -n "${NUM_RES:-}" ]] && SHAPE_ARGS="$SHAPE_ARGS --num-res $NUM_RES"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

say "TRAIN $TAG  (unet base=${BASE:-64} mult=${MULT:-1,2,4,4} num_res=${NUM_RES:-2} batch=${BATCH:-390} steps=${STEPS:-10000})"
"$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
  --backbone unet --size S --objective reg --residual 1 --native-lr "${NATIVE_LR:-1}" \
  $SHAPE_ARGS \
  --lr-patch 64 --batch "${BATCH:-390}" --amp ${COMPILE_FLAG:-} --num-workers "${WORKERS:-12}" \
  --cache-data 1 \
  --lr "${LR:-3e-4}" --warmup "${WARMUP:-500}" --steps "${STEPS:-10000}" \
  --eval-every "${EVAL_EVERY:-500}" --val-pairs 16 --patience 6 --min-steps "${MIN_STEPS:-1000}" \
  --val-eval-steps 8 --save-every 5000 \
  2>&1 | tee "$LOG/$TAG.train.log" | grep -E "PixelUNet|native_lr|sampler:|VAL |EARLY|Error|Traceback" | tail -60

say "EVAL $TAG (official 100 pairs)"
if [[ -f "$OUT/ckpt_best.pt" ]]; then
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/ckpt_best.pt" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 \
    --out "$OUT/eval_official" 2>&1 | tee "$LOG/$TAG.eval.log" | grep -E "OFFICIAL|FAIL" | tail -5
fi

say "$TAG DONE [$(ts)]"
