#!/usr/bin/env bash
# U-Net on RealSR, no pretraining -- the direct replacement for pixel_reg.
#
# Sizing came from bench_unet_batch.py:
#   UNet base=64 mult=(1,2,4,4) num_res=2 -> 18.67M params
#   batch 160 -> 38.8 GB peak (the 4090 goes super-linear past ~40 GB)
#   compiled 0.423 s/step -> 379 samples/s  (DiT-S b=28: 70 samples/s, 5.4x less)
#
# Sample budget: pixel_reg saw 16000 x 28 = 448k samples.  At batch 160 that is
# only 2800 steps, so 20000 steps gives 3.2M samples -- 7x the DiT's budget, and
# it still costs under 2.5 h.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion/unet_reg"
LOG="$ROOT/experiments/diffusion/logs"
mkdir -p "$OUT" "$LOG"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

say "TRAIN unet_reg (U-Net base=64 18.7M, objective=reg, RealSR only, batch 160)"
"$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
  --backbone unet --size S --objective reg --residual 1 --native-lr 1 \
  --lr-patch 64 --batch "${BATCH:-390}" --amp ${COMPILE_FLAG:-} --num-workers "${WORKERS:-12}" \
  --cache-data 1 \
  --lr "${LR:-3e-4}" --warmup "${WARMUP:-500}" --steps "${STEPS:-20000}" \
  --eval-every "${EVAL_EVERY:-500}" --val-pairs 16 --patience 6 --min-steps "${MIN_STEPS:-1000}" \
  --val-eval-steps 8 --save-every 5000 \
  2>&1 | tee "$LOG/unet_reg.train.log" | grep -E "PixelUNet|compile|VAL |EARLY|Error|Traceback" | tail -60

say "EVAL unet_reg (official 100 pairs)"
if [[ -f "$OUT/ckpt_best.pt" ]]; then
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/ckpt_best.pt" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 \
    --out "$OUT/eval_official" 2>&1 | tee "$LOG/unet_reg.eval.log" | grep -E "OFFICIAL|FAIL" | tail -5
fi

say "UNET REG DONE [$(ts)]"
