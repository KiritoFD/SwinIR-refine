#!/usr/bin/env bash
# Channel-shape sweep at stride 1, no pretraining.
#
# The axis is "more activation, higher memory-per-FLOP ratio".  Activations go as
# sum(C*s) and FLOPs as sum(C^2*s), so flattening the channel pyramid buys ratio
# but costs activation count -- which is why each shape gets its own base to keep
# the activation count at or above the current 1901k.
#
# (1,1,1,1) is deliberately left out: flattening the pyramid entirely means there
# is no pyramid, which stops being a U-Net.
#
# Every run: 10000 steps, batch sized to ~35 GB, then the official 100-pair eval
# with MUSIQ/MANIQA.  Steps are saturated (8000 -> 11000 was +0.01 dB) so 10000
# is enough and the runs stay short.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/shape_sweep}"
STEPS="${STEPS:-10000}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

# name:base:mult:batch
RUNS="${RUNS:-
1122_b96:96:1,1,2,2:96
1124_b80:80:1,1,2,4:128
1124_b96:96:1,1,2,4:96
1244_b80:80:1,2,4,4:96
1122_b80:80:1,1,2,2:128
}"

T0=$(date +%s)
for line in $RUNS; do
  [ -z "$line" ] && continue
  IFS=: read -r NAME BASE MULT BATCH <<< "$line"
  OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then
    say "SKIP $NAME -- already done"
    continue
  fi

  # the card does not hand VRAM back instantly, and launching into a stale
  # allocation OOMs on step 0 and looks like a hang
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && break
    sleep 10
  done

  say "TRAIN $NAME  base=$BASE mult=$MULT batch=$BATCH ($(( STEPS * BATCH )) samples)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base "$BASE" --mult "$MULT" --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch "$BATCH" --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps "$STEPS" \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 5000 > "$LOG/shape_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|native_lr|VAL |EARLY|Error|Traceback" "$LOG/shape_${NAME}.train.log" | tail -25

  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/ckpt_best.pt" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/shape_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/shape_${NAME}.eval.log" | tail -3
done

say "SHAPE SWEEP DONE  total $(( ($(date +%s) - T0) / 60 )) min"
