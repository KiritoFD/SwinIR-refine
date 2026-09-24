#!/usr/bin/env bash
# ~48h one-factor-at-a-time campaign on top of the (soon) champion recipe:
#   init=BSRGAN-pretrained b64_ft15k + Muon(5e-3) + wavelet-HF lambda8 (zero... this is pretrained).
# Waits for the champmuon run to finish, then sweeps remaining knobs -- all EXISTING
# train_pixel flags, no new code.  Sample budget pinned at ~1.28M per arm (batch x steps)
# so every row is comparable to the champion.  Judge by SSIM/MUSIQ/MANIQA (PSNR secondary),
# +-0.05 = tie.  No grad-ckpt (computation-bound); memory handled by batch.
# Wavelet-loss knobs first (the winning mechanism -> highest prior value), then optimizer,
# then architecture.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
BASE_EVAL="$ROOT/experiments/diffusion/stack_dwt/b64_pre_dwt8_muon/eval_iqa/eval.json"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="$ROOT/experiments/diffusion/sweep48"
mkdir -p "$OUT_ROOT" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

# base recipe (champion knobs); arms append overrides (argparse: last flag wins)
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --steps 10000 --ema 0.999 \
  --init $CHAMP --optimizer muon --muon-lr 5e-3 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"

echo "[sweep48 $(date +%m-%d\ %H:%M:%S)] waiting for champmuon eval ($BASE_EVAL)..."
while [ ! -f "$BASE_EVAL" ]; do
  tmux has-session -t champmuon 2>/dev/null || { [ -f "$BASE_EVAL" ] || sleep 20; }
  [ -f "$BASE_EVAL" ] && break
  sleep 30
done
wait_gpu
echo "[sweep48 $(date +%H:%M:%S)] base ready -> starting campaign"

# name | batch | steps | extra-flags
ARMS="
s_dwt4       | 128 | 10000 | --dwt-weight 4
s_dwt6       | 128 | 10000 | --dwt-weight 6
s_dwt12      | 128 | 10000 | --dwt-weight 12
s_dwt16      | 128 | 10000 | --dwt-weight 16
s_lv1        | 128 | 10000 | --dwt-levels 1
s_lv3        | 128 | 10000 | --dwt-levels 3
s_db2        | 128 | 10000 | --dwt-basis db2
s_smoothl1   | 128 | 10000 | --reg-loss smoothl1
s_mlr3       | 128 | 10000 | --muon-lr 3e-3
s_mlr7       | 128 | 10000 | --muon-lr 7e-3
s_wd1e2      | 128 | 10000 | --weight-decay 1e-2
s_ema9999    | 128 | 10000 | --ema 0.9999
s_warm1000   | 128 | 10000 | --warmup 1000
s_steps20k   | 128 | 20000 | --min-steps 5000
s_base96     |  96 | 13333 | --base 96
s_base128    |  64 | 20000 | --base 128 --min-steps 5000
s_nres3      |  96 | 13333 | --num-res 3
s_attn123    |  64 | 20000 | --attn-levels 1,2,3 --min-steps 5000
s_mult1248   |  96 | 13333 | --mult 1,2,4,8
s_patch80    |  64 | 20000 | --lr-patch 80 --min-steps 5000
"

run_arm() {
  local NAME="$1" BATCH="$2" STEPS="$3"; shift 3
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME"; return 0; fi
  wait_gpu
  say "TRAIN $NAME (batch=$BATCH steps=$STEPS | $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE --batch "$BATCH" --steps "$STEPS" "$@" > "$LOG/s48_${NAME}.train.log" 2>&1
  grep -E "optimizer=|VAL |EARLY|done|Error|Traceback|nan" "$LOG/s48_${NAME}.train.log" | tail -12
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/s48_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/s48_${NAME}.eval.log" | tail -2
}

echo "$ARMS" | while IFS='|' read -r NAME BATCH STEPS FLAGS; do
  [ -z "$(echo "$NAME" | tr -d ' ')" ] && continue
  NAME=$(echo "$NAME" | tr -d ' '); BATCH=$(echo "$BATCH" | tr -d ' '); STEPS=$(echo "$STEPS" | tr -d ' ')
  run_arm "$NAME" "$BATCH" "$STEPS" $FLAGS
done
say "SWEEP48 DONE"
