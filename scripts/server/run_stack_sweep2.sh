#!/usr/bin/env bash
# Round 5: find the perception PEAK of dwt-loss on the pretrained champion.
# Round 4 showed lambda 1->5 monotonically raising SSIM/MANIQA/PSNR-Y with MUSIQ
# flat (~55.77-55.81); lambda=5 is the current best (SSIM .9270 / MANIQA .3506 /
# MUSIQ 55.77 / Y 34.267).  We must NOT extrapolate blindly -- too-high lambda can
# over-sharpen and hurt perception (the FFN/froute failure mode).  Probe lambda 8
# and 10 on init=b64_ft15k to locate where SSIM/MANIQA peak or MUSIQ turns over.
set -uo pipefail
ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
CHAMP="${CHAMP:-$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/stack_dwt}"
mkdir -p "$OUT_ROOT" "$LOG"
[ -f "$CHAMP" ] || { echo "FATAL: champion ckpt not found: $CHAMP"; exit 1; }
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
  say "TRAIN $NAME  (init=$CHAMP extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE --init "$CHAMP" "$@" > "$LOG/sd_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|new-arms|init weights|VAL |EARLY|Error|Traceback" "$LOG/sd_${NAME}.train.log" | tail -20
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/sd_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/sd_${NAME}.eval.log" | tail -3
}
T0=$(date +%s)
train_and_eval b64_pre_dwt8  --dwt-loss --dwt-weight 8  --dwt-levels 2
train_and_eval b64_pre_dwt10 --dwt-loss --dwt-weight 10 --dwt-levels 2
say "STACK LAMBDA HIGH SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
