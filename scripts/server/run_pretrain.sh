#!/usr/bin/env bash
# SwinIR/BSRGAN-style pretraining, then fine-tune on RealSR.
#
# Why: our 34M DiT is trained on only 406 RealSR pairs, while the published
# SwinIR/BSRGAN models are trained on ~14-20k images with a synthetic
# high-order degradation. This script reproduces that *data recipe* (not the
# architecture) so we can separate "DiT is bad" from "DiT is starved".
#
#   stage 1  pretrain  DiT-S/reg on BSRGAN-degraded DIV2K(+Flickr2K)
#            val stays the RealSR held-out split, so we watch zero-shot transfer
#   stage 2  fine-tune the stage-1 ckpt on RealSR Train (weights only, step reset)
#   stage 3  official 100-pair eval of the fine-tuned model
set -uo pipefail

export ROOT="${ROOT:-/home/ds/realsr}"
export PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
export DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/pretrain"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

# pick every pretraining HR folder that exists
HR=""
for d in data/pretrain/DIV2K_train_HR data/pretrain/DIV2K_valid_HR data/pretrain/Flickr2K; do
  [[ -d "$d" ]] && HR="${HR:+$HR,}$d"
done
if [[ -z "$HR" ]]; then
  echo "no pretraining HR data under data/pretrain/ — run scripts/server/download_pretrain.sh first"
  exit 1
fi

SIZE="${SIZE:-S}"
STEPS1="${STEPS1:-30000}"
STEPS2="${STEPS2:-16000}"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

say "PRETRAIN data: $HR"

# ------------------------------------------------ stage 1: synthetic pretrain
"$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/pt_pixel_reg" \
  --pretrain-root "$HR" --size "$SIZE" --objective reg --steps "$STEPS1" \
  --lr-patch 64 --batch 28 --amp --num-workers 16 --compile \
  --eval-every 1000 --val-pairs 16 --patience 100 --min-steps 100000 \
  --val-eval-steps 8 --save-every 5000 \
  2>&1 | tee "$LOG/pt_pixel_reg.train.log" | grep -E "PixelDiT|compile|pretrain|VAL |EARLY|Error|Traceback" | tail -40

# ------------------------------------------------ stage 2: fine-tune on RealSR
if [[ -f "$OUT/pt_pixel_reg/ckpt_best.pt" ]]; then
  say "FINETUNE from stage 1"
  "$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/ft_pixel_reg" \
    --init "$OUT/pt_pixel_reg/ckpt_best.pt" \
    --size "$SIZE" --objective reg --steps "$STEPS2" \
    --lr-patch 64 --batch 28 --amp --num-workers 8 --compile \
    --eval-every 1000 --val-pairs 16 --patience 8 --min-steps 6000 \
    --val-eval-steps 8 --save-every 5000 \
    2>&1 | tee "$LOG/ft_pixel_reg.train.log" | grep -E "PixelDiT|init weights|VAL |EARLY|Error|Traceback" | tail -40
else
  echo "  no stage-1 ckpt, skipping fine-tune"
fi

# ------------------------------------------------ stage 3: official eval
for tag in ft_pixel_reg; do
  [[ -f "$OUT/$tag/ckpt_best.pt" ]] || continue
  say "EVAL $tag"
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/$tag/ckpt_best.pt" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 8 --steps 8 \
    --out "$OUT/$tag/eval_official" 2>&1 | tee "$LOG/$tag.eval.log" | grep -E "OFFICIAL|FAIL" | tail -5
done

say "PRETRAIN DONE [$(ts)]"
