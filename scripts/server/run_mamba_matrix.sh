#!/usr/bin/env bash
# Mamba matrix: every arm that can reach a meaningful sample budget, plus the
# controls needed to read it.
#
# The point of the design is to separate three things that got conflated in the
# first Mamba run:
#   * architecture (scan vs conv)
#   * stride (stride-1 vs the stride-2 stem the first run had to use)
#   * training budget (the heavy configs are so slow they never got enough samples)
#
# So: each Mamba arm gets a fixed wall-clock budget; the light stride-1 arm is
# matched against a stride-2 arm at the SAME sample count; and the U-Net anchor is
# re-run at that same sample count, because comparing a 329k-sample Mamba against
# a 960k-sample U-Net would measure the budget, not the architecture.
#
# All arms: reg objective, residual, mamba_ssm backend, grad-ckpt, lr 3e-4 cosine.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/diffusion/mamba_matrix}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

# name:scale:window:dim:groups:blocks:expand:state:batch:steps
ARMS="${ARMS:-
m_s1_c96:1:32:96:2:3:1:8:128:2100
m_s2_c96_match:2:0:96:2:3:1:8:512:640
m_s2_c96_full:2:0:96:2:3:1:8:512:3000
m_s2_c128_e1:2:0:128:4:4:1:8:512:1400
m_s2_c128_e2:2:0:128:4:4:2:8:320:560
}"

T0=$(date +%s)
for line in $ARMS; do
  [ -z "$line" ] && continue
  IFS=: read -r NAME SC W DIM G B E ST BATCH STEPS <<< "$line"
  OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then
    say "SKIP $NAME -- done"
    continue
  fi
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && break
    sleep 10
  done

  WINFLAG=""
  [ "$W" -gt 0 ] 2>/dev/null && WINFLAG="--ssm-shift"

  say "TRAIN $NAME scale=$SC win=$W dim=$DIM ${G}x${B} e=$E st=$ST batch=$BATCH steps=$STEPS ($(( STEPS * BATCH )) samples)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone mamba --objective reg --residual 1 \
    --base "$DIM" --num-groups "$G" --num-res "$B" \
    --ssm-state "$ST" --ssm-expand "$E" --ssm-scale "$SC" --ssm-window "$W" \
    $WINFLAG --ssm-backend mamba_ssm --grad-ckpt \
    --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch "$BATCH" --amp --num-workers 8 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps "$STEPS" \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 300 \
    --val-eval-steps 8 --save-every 1000 > "$LOG/mm_${NAME}.train.log" 2>&1
  grep -E "mamba dim|^step|VAL |EARLY|Error|OutOfMemory" "$LOG/mm_${NAME}.train.log" | tail -8

  CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/mm_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mm_${NAME}.eval.log" | tail -3
done

# ---- controls: the U-Net anchor at the sample budgets the Mambas reached ----
for spec in "u_b64_329k:3420:96" "u_b64_188k:1960:96" "u_b64_439k:4570:96"; do
  IFS=: read -r NAME STEPS BATCH <<< "$spec"
  OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then
    say "SKIP $NAME -- done"
    continue
  fi
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && break
    sleep 10
  done
  say "TRAIN CONTROL $NAME ($(( STEPS * BATCH )) samples)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch "$BATCH" --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps "$STEPS" \
    --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 300 \
    --val-eval-steps 8 --save-every 1000 > "$LOG/mm_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|^step|VAL |EARLY|Error" "$LOG/mm_${NAME}.train.log" | tail -5
  CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/mm_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mm_${NAME}.eval.log" | tail -3
done

say "MAMBA MATRIX DONE  total $(( ($(date +%s) - T0) / 60 )) min"
