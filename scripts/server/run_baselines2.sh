#!/usr/bin/env bash
# Second baseline batch (SRResNet, RRDB) — waits for the first chain (baserep,
# currently training RCAN) to finish so we never run two jobs on the saturated GPU.
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG=experiments/diffusion/logs; OUT=experiments/diffusion/baselines
mkdir -p "$LOG" "$OUT"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
echo "[baserep2 $(date +%H:%M)] waiting for baserep to finish..."
while tmux has-session -t baserep 2>/dev/null; do sleep 30; done
wait_gpu
run() {
  local B="$1"
  local DIR="$OUT/${B}_x2"
  if [ -f "$DIR/eval_iqa/eval.json" ]; then say "SKIP $B"; return 0; fi
  wait_gpu; say "TRAIN $B (x2, RealSR Train, from scratch, L1)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" \
    --backbone "$B" --size S --base 64 --scale 2 --objective reg --lr-patch 64 \
    --batch 64 --steps 20000 --lr 3e-4 --warmup 500 --ema 0.999 --amp --num-workers 12 \
    --cache-data 0 --eval-every 1000 --val-pairs 16 --patience 20 --min-steps 4000 \
    --val-eval-steps 8 --save-every 5000 > "$LOG/bl_${B}.train.log" 2>&1
  grep -E "align=|done|Error|Traceback|nan|out of memory" "$LOG/bl_${B}.train.log" | tail -8
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  say "EVAL $B (A-strict + IQA + TTA)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 8 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/bl_${B}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/bl_${B}.eval.log" | tail -2
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 8 --tile-batch 16 --steps 8 --iqa --tta --out "$DIR/eval_iqa_tta" > "$LOG/bl_${B}.tta.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/bl_${B}.tta.log" | tail -2
}
run srresnet
run rrdb
say "BASELINES-2 DONE"
