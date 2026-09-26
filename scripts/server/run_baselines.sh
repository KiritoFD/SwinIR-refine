#!/usr/bin/env bash
# Reproduce classic SR baselines on the SAME protocol as our model (A-strict:
# train RGB on RealSR Train, test Limited-range Y / no shave / 100 pairs), instead
# of quoting paper numbers from a different protocol.  EDSR and RCAN are the
# pre-upsampling variants (see diffusion/baselines_sr.py docstring) — honest label.
# Sample budget matched to the s1_b64 anchor (batch*steps = 1.28M).
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG=experiments/diffusion/logs; OUT=experiments/diffusion/baselines
mkdir -p "$LOG" "$OUT"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

run() {  # $1 backbone
  local B="$1"
  local DIR="$OUT/${B}_x2"
  if [ -f "$DIR/eval_iqa/eval.json" ]; then say "SKIP $B"; return 0; fi
  wait_gpu; say "TRAIN $B (x2, RealSR Train, from scratch, L1)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" \
    --backbone "$B" --size S --base 64 --scale 2 --objective reg --lr-patch 64 \
    --batch 64 --steps 20000 --lr 3e-4 --warmup 500 --ema 0.999 --amp --num-workers 12 \
    --cache-data 0 --eval-every 1000 --val-pairs 16 --patience 20 --min-steps 4000 \
    --val-eval-steps 8 --save-every 5000 > "$LOG/bl_${B}.train.log" 2>&1
  grep -E "align=|VAL |done|Error|Traceback|nan|CUDA out of memory" "$LOG/bl_${B}.train.log" | tail -10
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  say "EVAL $B (A-strict + IQA + TTA)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 8 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/bl_${B}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/bl_${B}.eval.log" | tail -2
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 8 --tile-batch 16 --steps 8 --iqa --tta --out "$DIR/eval_iqa_tta" > "$LOG/bl_${B}.tta.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/bl_${B}.tta.log" | tail -2
}
run edsr
run rcan
say "BASELINES DONE"
