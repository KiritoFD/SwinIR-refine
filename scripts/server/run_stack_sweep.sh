#!/usr/bin/env bash
# Round 4: nail the perception-optimal lambda ON TOP of the pretrained champion.
# Re-judged by SSIM/MUSIQ/MANIQA (PSNR secondary): pretrained+dwt-lambda3 is the new
# best (SSIM .9266 / MUSIQ 55.77 / MANIQA .3501).  Zero-pretrain showed lambda=5 was
# the all-metrics peak, so probe lambda 2 and 5 on the pretrained base to find where
# MUSIQ/MANIQA peak (and whether higher lambda trades SSIM for MUSIQ).
#   b64_pre_dwt2  = init b64_ft15k + dwt-loss lambda2
#   b64_pre_dwt5  = init b64_ft15k + dwt-loss lambda5
# Same b64 anchor recipe (native-lr 0), batch 128, no grad-ckpt (computation-bound).
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
train_and_eval b64_pre_dwt2 --dwt-loss --dwt-weight 2 --dwt-levels 2
train_and_eval b64_pre_dwt5 --dwt-loss --dwt-weight 5 --dwt-levels 2
say "STACK LAMBDA SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
