#!/usr/bin/env bash
# Direction C run: train the residual HF rectified-flow head on the frozen regression
# champion (b64_pre_dwt8), then eval the refined SR.  Judge by MUSIQ/MANIQA (PSNR
# secondary): does generating the HF residual lift perception at a small/neutral PSNR
# cost, where full-image flow previously collapsed below the bicubic floor?
#   baseline (regression only): SSIM .9271 / MUSIQ 55.87 / MANIQA .3513 / Y 34.278
# Two evals: full-scale refine and a gentler scale, at 4 NFE.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
REG="$ROOT/experiments/diffusion/stack_dwt/b64_pre_dwt8/ckpt_best.pt"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
HF="$ROOT/experiments/diffusion/hf_flow"
mkdir -p "$HF" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 120); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

HEAD="$HF/hf_x2"
# v1 trained the head WITH attention -> OOM on full-image refine, and evaluated at
# scale>=0.6 which is far too aggressive for a from-noise HF residual. Purge v1 so the
# conv-only head retrains and the small-scale evals are fresh.
if [ -f "$HEAD/refine_s1.0/eval_iqa/eval.json" ]; then rm -rf "$HEAD"; fi
if [ ! -f "$HEAD/ckpt_best.pt" ] && [ ! -f "$HEAD/ckpt_last.pt" ]; then
  wait_gpu
  say "TRAIN hf head (frozen b64_pre_dwt8 -> HF residual flow)"
  "$PY" -u -m diffusion.train_hf_flow --data-root "$DATA" --reg-ckpt "$REG" --out "$HEAD" \
    --head-base 64 --head-mult 1,2,4 --head-res 2 --batch 64 --steps 8000 --lr 2e-4 \
    --warmup 300 --ema 0.999 --amp 1 --eval-every 500 --val-pairs 16 --save-every 2000 \
    --num-workers 12 > "$LOG/hf_x2.train.log" 2>&1
  grep -E "hf head|flowL|VALFLOW|done|Error|Traceback" "$LOG/hf_x2.train.log" | tail -20
fi
CK="$HEAD/ckpt_best.pt"; [ -f "$CK" ] || CK="$HEAD/ckpt_last.pt"

for spec in "s0.10 2 0.10" "s0.20 2 0.20" "s0.35 2 0.35"; do
  set -- $spec; NAME="$1"; NSTEPS="$2"; SCALE="$3"
  OUT="$HEAD/refine_$NAME"
  [ -f "$OUT/eval_iqa/eval.json" ] && { say "SKIP refine_$NAME"; continue; }
  wait_gpu
  say "EVAL refine $NAME (hf-steps=$NSTEPS hf-scale=$SCALE)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$REG" \
    --mode pixel --objective reg --scale 2 --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --hf-head "$CK" --hf-steps "$NSTEPS" --hf-scale "$SCALE" \
    --out "$OUT/eval_iqa" > "$LOG/hf_refine_$NAME.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/hf_refine_$NAME.eval.log" | tail -2
done
say "HF FLOW DONE"
