#!/usr/bin/env bash
# Diagnostic: did pixel_flow actually get worse, or did we just measure it at
# too low an NFE?
#
# The training VAL curve peaked at 26.25 (step 4000, end of warmup) and then
# fell to ~20 for the rest of the run.  Two explanations:
#   (a) training went unstable at peak LR -- fix = retrain with a lower LR
#   (b) as the fit sharpens the learned ODE gets stiffer, so a FIXED NFE=16
#       val looks worse even though the model is better -- fix = evaluate the
#       late checkpoint at a high NFE, no retrain needed
# They cost 3.3h vs 0.5h, so it is worth 20 minutes to tell them apart.
#
# Compares ckpt_best (step 4k) against ckpt_last (step 12k) at NFE 64.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
OUT="$ROOT/experiments/diffusion/pixel_flow"
LOG="$ROOT/experiments/diffusion/logs"
STEPS="${STEPS:-32}"
PAIRS="${PAIRS:-4}"
cd "$ROOT" || exit 1

for ck in ckpt_best ckpt_last; do
  [[ -f "$OUT/$ck.pt" ]] || { echo "missing $OUT/$ck.pt"; continue; }
  echo "===== $ck  steps=$STEPS (NFE $((STEPS*2)))  pairs=$PAIRS ====="
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/$ck.pt" --mode pixel --objective flow --residual \
    --tile 64 --pad 16 --tile-batch 4 --steps "$STEPS" \
    --max-pairs "$PAIRS" --out "$OUT/diag_${ck}_nfe$((STEPS*2))" \
    2>&1 | tee "$LOG/diag_${ck}_nfe$((STEPS*2)).log" | grep -E "OFFICIAL|FAIL"
done
echo "DIAG DONE $(date +%m-%d\ %H:%M:%S)"
