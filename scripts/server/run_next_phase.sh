#!/usr/bin/env bash
# Next phase (chained, unattended): waits for round-5 (tmux stacksweep2) to end,
# then, serially:
#   A) TTA self-ensemble eval on the two best x2 ckpts (free perception/PSNR boost,
#      eval-only, writes <dir>/eval_iqa_tta so the non-TTA numbers stay).
#   B) finer lambda sweep on the pretrained base (6, 7) to pin the perception peak.
#   C) x3 / x4 migration of the champion recipe (dwt-loss lambda5) + a plain control
#      at each scale, to (i) report scale-generalisation and (ii) confirm dwt-loss
#      helps at heavier degradation too.
# Computation-bound: no grad-ckpt.  scale3 uses lr-patch 48 (HR144, batch 96),
# scale4 uses lr-patch 32 (HR128, batch 128) to keep stride-1 activations in 40G.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
EXP="$ROOT/experiments/diffusion"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() {
  for _ in $(seq 1 120); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    sleep 15
  done
}

echo "[next $(date +%H:%M:%S)] waiting for round-5 (tmux stacksweep2)..."
while tmux has-session -t stacksweep2 2>/dev/null; do sleep 30; done
wait_gpu
echo "[next $(date +%H:%M:%S)] round-5 done; starting next phase"

# ---- A) TTA eval on the two best x2 ckpts ----------------------------------
eval_tta() {  # $1 = dir containing ckpt_best.pt
  local DIR="$1"
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  local OUT="$DIR/eval_iqa_tta"
  if [ -f "$OUT/eval.json" ]; then say "SKIP TTA $DIR -- done"; return 0; fi
  wait_gpu
  say "TTA EVAL $DIR"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg --tta \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT" > "$LOG/tta_$(basename $DIR).log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/tta_$(basename $DIR).log" | tail -3
}
eval_tta "$EXP/stack_dwt/b64_pre_dwt5"
eval_tta "$EXP/stack_dwt/b64_pre_dwt8"
eval_tta "$EXP/wave_arms/s1_b64_dwt_w5"

# ---- B) finer lambda sweep on the pretrained base ---------------------------
train_stack() {  # $1 name, $2 lambda
  local NAME="$1"; local LAM="$2"
  local OUT="$EXP/stack_dwt/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  (init champion + dwt lambda$LAM)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 10000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 2500 --init "$CHAMP" \
    --dwt-loss --dwt-weight "$LAM" --dwt-levels 2 > "$LOG/sd_${NAME}.train.log" 2>&1
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/sd_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/sd_${NAME}.eval.log" | tail -2
}
train_stack b64_pre_dwt6 6
train_stack b64_pre_dwt7 7

# ---- C) x3 / x4 migration: dwt lambda5 recipe + plain control ---------------
train_scale() {  # $1 name, $2 scale, $3 lr-patch, $4 batch, rest extra flags
  local NAME="$1" SC="$2" LP="$3" BS="$4"; shift 4
  local OUT="$EXP/scale_$SC/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME x$SC -- done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME scale=$SC lrpatch=$LP batch=$BS (extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --scale "$SC" --decoded-manifest data/decoded/manifest.json \
    --lr-patch "$LP" --batch "$BS" --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 10000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 2500 "$@" > "$LOG/sc_${NAME}_x${SC}.train.log" 2>&1
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME x$SC (official, non-TTA)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --scale "$SC" --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/sc_${NAME}_x${SC}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/sc_${NAME}_x${SC}.eval.log" | tail -2
}
# x3: HR144 (lr-patch48) batch96 ; x4: HR128 (lr-patch32) batch128
train_scale s1_b64_plain 3 48 96
train_scale s1_b64_dwt5  3 48 96 --dwt-loss --dwt-weight 5 --dwt-levels 2
train_scale s1_b64_plain 4 32 128
train_scale s1_b64_dwt5  4 32 128 --dwt-loss --dwt-weight 5 --dwt-levels 2

say "NEXT PHASE DONE"
