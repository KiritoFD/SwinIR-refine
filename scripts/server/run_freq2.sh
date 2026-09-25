#!/usr/bin/env bash
# Freq 2.0 experiments: does a better wavelet basis / anisotropic band weighting
# beat Haar at the current champion point (pretrained b64 + dwt lambda8)?
# Baseline to beat: b64_pre_dwt8 (haar lambda8) = SSIM .92709 / MUSIQ 55.87 /
# MANIQA .3513 / Y 34.278.
#   F1 b64_pre_dwt8_db2    : db2 basis
#   F2 b64_pre_dwt8_db4    : db4 basis
#   F3 b64_pre_dwt8_db2a   : db2 + anisotropic (HL 1.6 / LH 0.8 / HH 1.0) -- astigmatism
# Plus the scale-adaptive lambda that x3 asked for (x3 haar lambda5 over-sharpened):
#   F4 scale_3/s1_b64_dwt1 (zero-pretrain, lambda1)   } smaller lambda at x3, does
#   F5 scale_3/s1_b64_dwt2 (zero-pretrain, lambda2)   } MUSIQ climb back above plain?
# Computation-bound: no grad-ckpt.
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
run_eval() {  # $1 ckpt, $2 out, $3 scale
  local SC="${3:-2}"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" \
    --mode pixel --objective reg --scale "$SC" --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$2/eval_iqa" > "$LOG/fq_$(basename $2).eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/fq_$(basename $2).eval.log" | tail -2
}

# ---- F1-F3: bases + anisotropy on the pretrained lambda8 champion ------------
train_pre() {  # $1 name, rest = dwt flags
  local NAME="$1"; shift
  local OUT="$EXP/stack_dwt/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME (init champion, $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 10000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 2500 --init "$CHAMP" \
    --dwt-loss --dwt-weight 8 --dwt-levels 2 "$@" > "$LOG/fq_${NAME}.train.log" 2>&1
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  run_eval "$CK" "$OUT" 2
}
train_pre b64_pre_dwt8_db2  --dwt-basis db2
train_pre b64_pre_dwt8_db4  --dwt-basis db4
train_pre b64_pre_dwt8_db2a --dwt-basis db2 --dwt-w-hl 1.6 --dwt-w-lh 0.8 --dwt-w-hh 1.0

# ---- F4-F5: scale-adaptive lambda at x3 -------------------------------------
train_x3() {  # $1 name, $2 lambda
  local NAME="$1" LAM="$2"
  local OUT="$EXP/scale_3/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME x3 -- done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME scale3 lrpatch48 batch96 lambda$LAM"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --scale 3 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 48 --batch 96 --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 10000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
    --val-eval-steps 8 --save-every 2500 \
    --dwt-loss --dwt-weight "$LAM" --dwt-levels 2 > "$LOG/fq_${NAME}_x3.train.log" 2>&1
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  run_eval "$CK" "$OUT" 3
}
train_x3 s1_b64_dwt1 1
train_x3 s1_b64_dwt2 2

say "FREQ 2.0 DONE"
