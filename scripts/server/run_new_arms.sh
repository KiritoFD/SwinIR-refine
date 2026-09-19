#!/usr/bin/env bash
# A/B matrix for the three new arms, all against the s1_b64 anchor
# (zero-pretrain, official 100 pairs Y 34.1083 / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416):
#
#   N1 s1_b64_coord   -- absolute frame coordinates on the input (breaks conv
#                        translation invariance; lens degradation is non-stationary)
#   N2 s1_b64_froute  -- per-ResBlock soft frequency routing (texture conv vs
#                        narrow smooth branch, alpha = f(Sobel energy))
#   N3 adv            -- adversarial degradation mining pretrain (min-max over a
#                        differentiable PSF-bank + noise adversary) -> RealSR
#                        fine-tune.  Same 25000x128 sample budget as the BSRGAN
#                        pretrain that returned only +0.09 dB, so the comparison
#                        is apples-to-apples.
#
# Everything except the arm's flag is IDENTICAL to the b64 anchor recipe:
# base 64, mult 1,2,4,4, num-res 2, native-lr 0, lr-patch 64, batch 128,
# lr 3e-4 warmup 500, 10000 steps, EMA 0.999, amp, val 16 pairs patience 15.
#
# Idempotent: an arm with an existing eval_iqa/eval.json is skipped.
# Launch:  tmux new -d -s newarms 'bash scripts/server/run_new_arms.sh'
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/new_arms}"
mkdir -p "$OUT_ROOT" "$LOG"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-128}"
PRE_STEPS="${PRE_STEPS:-25000}"
PRE_BATCH="${PRE_BATCH:-128}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K}"
PRETRAIN_VAL_ROOT="${PRETRAIN_VAL_ROOT:-data/pretrain/DIV2K_valid_HR}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

wait_gpu() {
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    echo "  waiting for GPU to drain: ${M} MiB"; sleep 10
  done
}

# the shared anchor recipe (everything here matches the b64 sweep exactly)
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch $BATCH --amp --num-workers 12 --cache-data 0 \
  --lr 3e-4 --warmup 500 --steps $STEPS \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"

train_and_eval() {  # $1 name, $2 extra train flags
  local NAME="$1"; shift
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  (extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE "$@" > "$LOG/na_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|unet native_lr|VAL |EARLY|Error|Traceback" "$LOG/na_${NAME}.train.log" | tail -25
  local CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/na_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/na_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)

# ---- N1 / N2: single-flag A/Bs on the anchor recipe -------------------------
train_and_eval s1_b64_coord  --coord
train_and_eval s1_b64_froute --freq-route

# ---- N3: adversarial min-max pretrain -> RealSR fine-tune -------------------
ADV_OUT="$OUT_ROOT/adv_pretrain"
if [ ! -f "$ADV_OUT/ckpt_best.pt" ]; then
  wait_gpu
  say "PRETRAIN adv  ($(( PRE_STEPS * PRE_BATCH )) samples, min-max vs G_phi)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$ADV_OUT" \
    $RECIPE --adv-deg \
    --pretrain-root "$PRETRAIN_ROOT" --pretrain-val-root "$PRETRAIN_VAL_ROOT" \
    --steps "$PRE_STEPS" --batch "$PRE_BATCH" \
    > "$LOG/na_adv_pretrain.log" 2>&1
  grep -E "PixelUNet|adversary|VAL |EARLY|Error|Traceback" "$LOG/na_adv_pretrain.log" | tail -25
fi
if [ -f "$ADV_OUT/ckpt_best.pt" ] && [ ! -f "$OUT_ROOT/s1_b64_adv_ft/eval_iqa/eval.json" ]; then
  rm -f "$OUT_ROOT/.adv_ft_done"
  # fine-tune from the adversarially pretrained weights on RealSR (same recipe)
  train_and_eval s1_b64_adv_ft --init "$ADV_OUT/ckpt_best.pt"
fi

say "NEW ARMS DONE  total $(( ($(date +%s) - T0) / 60 )) min"
