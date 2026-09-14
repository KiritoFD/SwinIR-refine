#!/usr/bin/env bash
# Launch E13 Align+Wiener training in tmux (Windows Python via /mnt)
set -euo pipefail
ROOT="/mnt/g/RealSR"
PY="/mnt/c/Users/xy/AppData/Local/Programs/Python/Python312/python.exe"
OUT="/mnt/g/RealSR/experiments/improve/E13_align_wiener"
LOG="$OUT/tmux_train.log"
mkdir -p "$OUT"

tmux has-session -t e13 2>/dev/null && tmux kill-session -t e13 || true

tmux new-session -d -s e13 bash -lc "
  cd '$ROOT' &&
  '$PY' -m mod_swinir.train \
    --data-root 'G:\\RealSR\\data\\RealSR(V3)' \
    --out 'G:\\RealSR\\experiments\\improve\\E13_align_wiener' \
    --scale 2 --arch mod --model-size base \
    --batch-size 4 --grad-accum 2 --lr-patch 64 \
    --steps 12000 --lr 2e-4 --warmup 150 --num-workers 0 \
    --eval-every 1500 --eval-pairs 6 --save-every 3000 \
    --cameras Canon,Nikon --amp --amp-dtype fp16 \
    --l1-only --align-loss --no-kpn --ema 0.999 \
    > '$LOG' 2>&1
"

echo "tmux sessions:"
tmux ls
echo "log: $LOG"
