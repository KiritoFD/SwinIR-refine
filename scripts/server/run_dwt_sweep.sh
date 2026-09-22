#!/usr/bin/env bash
# Round 1b: complete the wavelet-loss lambda sweep on the plain b64 anchor.
# Round 1 gave lambda=1 -> 34.1895 and lambda=3 -> 34.2199 (new project best on
# PSNR-Y, beating the BSRGAN-pretrain 34.1988).  This fills the curve:
#   lambda=2 (between the two winners) and lambda=5 (top of the proposal range,
#   probes where extra high-freq weight starts costing PSNR / over-sharpening).
# Same b64 anchor recipe, native-lr 0, batch 128, no grad-ckpt (computation-bound).
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/wave_arms}"
mkdir -p "$OUT_ROOT" "$LOG"
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --lr 3e-4 --warmup 500 --steps 10000 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() {
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    sleep 10
  done
}
train_and_eval() {
  local NAME="$1"; shift
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  (extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE "$@" > "$LOG/wa_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|new-arms|VAL |EARLY|Error|Traceback" "$LOG/wa_${NAME}.train.log" | tail -20
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/wa_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/wa_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)
train_and_eval s1_b64_dwt_w2 --dwt-loss --dwt-weight 2 --dwt-levels 2
train_and_eval s1_b64_dwt_w5 --dwt-loss --dwt-weight 5 --dwt-levels 2
say "DWT LAMBDA SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
