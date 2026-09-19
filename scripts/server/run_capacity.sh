#!/usr/bin/env bash
# The "addition" step: keep stride 1 and the channel pyramid, widen the base to
# 128 (Tensor Core aligned), and push capacity into the 50-100M range.
#
#   1. b128        -- pure width scaling, 72.5M
#   2. b128 + FFN  -- same, plus a spatial-gated FFN in every residual block, 97.1M
#
# Both channels stay multiples of 128 all the way down ([128,256,512,512]), which
# is what the gated FFN's alignment assert demands and what keeps the GEMMs off
# the padded/kernel-downgraded path.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/capacity}"
STEPS="${STEPS:-10000}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

# name:base:mult:batch:ffn:extra_train_flags(semicolon separated, no spaces)
# The FFN variants cannot run train_pixel's validation: run_val does a single
# untiled forward over the whole 1000x1400 test image, which is ~2.7x the
# pixels of a training batch, and the gated FFN's intermediates blow past 47 GB
# there.  For those runs we skip validation and evaluate the last checkpoint.
RUNS="${RUNS:-
b128:128:1,2,4,4:64:0:
b128_ffn:128:1,2,4,4:32:1:--eval-every=0
}"

T0=$(date +%s)
for line in $RUNS; do
  [ -z "$line" ] && continue
  IFS=: read -r NAME BASE MULT BATCH FFN EXTRA <<< "$line"
  OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then
    say "SKIP $NAME -- already done"
    continue
  fi

  # the card does not hand VRAM back instantly
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && break
    sleep 10
  done

  FFNFLAG=""
  [ "$FFN" = "1" ] && FFNFLAG="--ffn"

  say "TRAIN $NAME  base=$BASE mult=$MULT batch=$BATCH ffn=$FFN ($(( STEPS * BATCH )) samples)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base "$BASE" --mult "$MULT" --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    $FFNFLAG \
    --lr-patch 64 --batch "$BATCH" --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps "$STEPS" \
    ${EXTRA//;/ } --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 2500 > "$LOG/cap_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|unet native_lr|VAL |EARLY|Error|Traceback" "$LOG/cap_${NAME}.train.log" | tail -25

  CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/cap_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/cap_${NAME}.eval.log" | tail -3
done

say "CAPACITY DONE  total $(( ($(date +%s) - T0) / 60 )) min"
