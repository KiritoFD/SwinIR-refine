#!/usr/bin/env bash
# RealSR post-training on top of the BSRGAN-pretrained b64.
#
# The chain's post-training only reached step 8000 before the 16-pair val early
# stopper fired, so this runs it properly: 15000 steps, and the patience is
# loosened so a noisy 16-pair plateau cannot cut it short again.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

CKPT="${CKPT:-$ROOT/experiments/diffusion/s1_pretrain_b64/b64_pretrain/ckpt_best.pt}"
OUT="${OUT:-$ROOT/experiments/diffusion/b64_ft15k}"
STEPS="${STEPS:-15000}"

for _ in $(seq 1 60); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( M < 2000 )) && break
  say "waiting for GPU to drain: ${M} MiB"
  sleep 10
done

if [ ! -f "$CKPT" ]; then
  say "NO PRETRAINED CKPT at $CKPT -- training from scratch instead"
  CKPT=""
else
  say "init from $CKPT"
fi
INIT=""
[ -n "$CKPT" ] && INIT="--init $CKPT"

say "POST-TRAIN b64 on RealSR ($(( STEPS * 128 )) samples, batch 128)"
"$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
  --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  $INIT \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --lr "${LR:-3e-4}" --warmup 500 --steps "$STEPS" \
  --eval-every 500 --val-pairs 16 --patience "${PATIENCE:-15}" --min-steps 3000 \
  --val-eval-steps 8 --save-every 5000 > "$LOG/b64_ft15k.train.log" 2>&1
grep -E "PixelUNet|native_lr|VAL |EARLY|Error|Traceback" "$LOG/b64_ft15k.train.log" | tail -40

say "EVAL b64_ft15k (official 100 pairs + MUSIQ/MANIQA)"
"$PY" -u -m diffusion.eval_official --data-root "$DATA" \
  --ckpt "$OUT/ckpt_best.pt" --mode pixel --objective reg \
  --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
  --out "$OUT/eval_iqa" > "$LOG/b64_ft15k.eval.log" 2>&1
grep -E "OFFICIAL|FAIL|Error" "$LOG/b64_ft15k.eval.log" | tail -5

say "POST-TRAIN DONE"
