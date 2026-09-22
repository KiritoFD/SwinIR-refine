#!/usr/bin/env bash
# Wavelet + Equivalence arms: the two "quick A/B" mechanisms that need NO
# architecture change (both are training-objective changes on the b64 anchor),
# run serially against the s1_b64 zero-pretrain anchor.
#
#   anchor  s1_b64               Y 34.1083 / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416
#   W1/W2   --dwt-loss           orthogonal Haar high-frequency subband L1 (fight
#                                L1 over-smoothing).  Zero inference cost; judge on
#                                MUSIQ/MANIQA + edge sharpness, PSNR may tie.
#   W3      --equiv              D4 group-equivariance self-supervision: a second
#                                forward on a flipped/rot90'd input, penalising
#                                |T(f(x))-f(T(x))|.  The two forward graphs are held
#                                together until backward (~2x activation).  We are
#                                COMPUTATION-bound, so NO --grad-ckpt (its recompute
#                                is pure loss here); the batch is halved to 64 instead
#                                (steps doubled to 20000, sample budget held at 1.28M).
#
# Everything except the arm's flag is IDENTICAL to the b64 anchor recipe
# (base 64, mult 1,2,4,4, num-res 2, native-lr 0, lr-patch 64, batch 128,
#  lr 3e-4 warmup 500, 10000 steps, EMA 0.999, amp, val 16, patience 15).
#
# Discipline (learned the hard way, FFN/coord/froute): 16-pair val is trend only;
# life-or-death is the official 100 pairs + IQA; within +-0.05 dB is a TIE.
#
# Idempotent: an arm with an existing eval_iqa/eval.json is skipped.
# Launch:  tmux new -d -s wavearms 'bash scripts/server/run_wave_arms.sh'
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
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-128}"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

wait_gpu() {
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    echo "  waiting for GPU to drain: ${M} MiB"; sleep 10
  done
}

RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch $BATCH --amp --num-workers 12 --cache-data 0 \
  --lr 3e-4 --warmup 500 --steps $STEPS \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"

train_and_eval() {  # $1 name, rest = extra train flags
  local NAME="$1"; shift
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  (extra: $*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    $RECIPE "$@" > "$LOG/wa_${NAME}.train.log" 2>&1
  grep -E "PixelUNet|new-arms|unet native_lr|VAL |EARLY|Error|Traceback" "$LOG/wa_${NAME}.train.log" | tail -25
  local CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/wa_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/wa_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)

# ---- W1/W2: wavelet high-frequency loss (two lambdas; backbone-agnostic) ----
train_and_eval s1_b64_dwt_w1 --dwt-loss --dwt-weight 1 --dwt-levels 2
train_and_eval s1_b64_dwt_w3 --dwt-loss --dwt-weight 3 --dwt-levels 2

# ---- W3: D4 equivariance self-supervision.  No grad-ckpt (computation-bound);
#      batch halved to fit the 2nd forward graph, steps x2 to keep 1.28M samples.
train_and_eval s1_b64_equiv --equiv --equiv-weight 0.25 --batch 64 --steps 20000

say "WAVE+EQUIV ARMS DONE  total $(( ($(date +%s) - T0) / 60 )) min"
