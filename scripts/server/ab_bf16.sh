#!/usr/bin/env bash
# A/B the new --bf16 eval path against the fp32 path it replaces.
#
# Every number in the current 2x2 table was produced in fp32 (eval_official had
# no autocast at all, unlike every other module in the repo).  Before switching
# the remaining evals to bf16 we need to know how much the metric moves.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
CKPT="${CKPT:-$ROOT/experiments/diffusion/pixel_reg/ckpt_best.pt}"
N="${N:-8}"
cd "$ROOT" || exit 1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

run() {
  local tag=$1; shift
  echo "===== $tag ====="
  /usr/bin/time -f "  wall %e s   peak_rss %M KB" \
    "$PY" -m diffusion.eval_official --data-root "$DATA" --ckpt "$CKPT" \
      --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 8 --steps 8 \
      --max-pairs "$N" "$@" --out "/tmp/ab_$tag" 2>&1 | \
    grep -E "OFFICIAL|wall |peak|FAIL"
}

run fp32
run bf16 --bf16
echo "AB DONE $(date +%m-%d\ %H:%M:%S)"
