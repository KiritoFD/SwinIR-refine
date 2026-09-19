#!/usr/bin/env bash
# SERIAL 12-hour plan: one training at a time, each sized to fill ~40 GB.
#
#   Phase 1 (t=0)     M1 mamba_c128  (2.75M, scale-2 scan, 4RG x 4 VSS)
#                     batch probed to fill ~40 GB (384 -> 320 -> 256 -> 192),
#                     2000 steps = 768k samples        wall ~6 h + eval
#   Phase 2 (t~6.3h)  M2 mamba_c256  (10.48M, same topology)
#                     batch probed likewise, 1800 steps = ~346k samples
#                                                      wall ~5.5 h + eval
#   ETA total ~12 h.  NO U-NET arms (the N3 finetune stays cancelled; its
#   question remains open with ckpt_best kept on disk).
#
# Why batch fills VRAM instead of lanes: user directive -- serial, card full.
# Sample throughput is ~batch-invariant (36 samples/s at c128), so the big
# batch costs nothing and buys the fill.  lr stays at the anchor's 3e-4.
#
# Launch:  tmux new -d -s plan12 'bash scripts/server/run_12h_plan.sh'
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null

LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="$ROOT/experiments/diffusion/plan12"
mkdir -p "$OUT_ROOT" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }

wait_gpu() {
  for _ in $(seq 1 120); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( M < 2000 )) && return 0
    echo "  waiting for GPU to drain: ${M} MiB"; sleep 15
  done
}

probe_batch() {  # $1 base dim, $2 scale, $3 candidate batches (big -> small)
  # stdout carries ONLY the chosen batch (this function runs in $(...));
  # every diagnostic goes to stderr
  for B in $3; do
    say "PROBE base=$1 scale=$2 batch=$B (8 steps)" >&2
    rm -rf "$OUT_ROOT/probe_$1_$B"
    if "$PY" -u -m diffusion.train_pixel --data-root "$DATA" \
        --out "$OUT_ROOT/probe_$1_$B" --backbone mamba --objective reg --residual 1 \
        --base "$1" --num-groups 4 --num-res 4 --ssm-state 16 --ssm-expand 2 \
        --ssm-scale "$2" --ssm-backend mamba_ssm --grad-ckpt \
        --lr-patch 64 --batch "$B" --steps 8 --eval-every 0 --num-workers 8 \
        --cache-data 0 --decoded-manifest data/decoded/manifest.json --amp \
        > "$LOG/p12_probe_b$B.log" 2>&1; then
      if ! grep -q "OutOfMemory" "$LOG/p12_probe_b$B.log"; then
        grep -E "step 000" "$LOG/p12_probe_b$B.log" | tail -2 >&2
        echo "$B"; return 0
      fi
    fi
    echo "  batch $B failed (OOM) -- trying next" >&2
  done
  echo "128"
}

train_and_eval() {  # $1 name, $2 batch, $3 steps, rest = model flags
  local NAME="$1" BATCH="$2" STEPS="$3"; shift 3
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME -- already done"; return 0; fi
  wait_gpu
  say "TRAIN $NAME  batch=$BATCH steps=$STEPS"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" \
    --backbone mamba --objective reg --residual 1 \
    --num-groups 4 --num-res 4 --ssm-state 16 --ssm-expand 2 \
    --ssm-scale 2 --ssm-backend mamba_ssm --grad-ckpt \
    --lr-patch 64 --batch "$BATCH" --steps "$STEPS" \
    --lr 3e-4 --warmup 500 --amp --num-workers 8 --cache-data 0 \
    --decoded-manifest data/decoded/manifest.json \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 1000 \
    --val-eval-steps 8 --save-every 500 "$@" \
    > "$LOG/p12_${NAME}.train.log" 2>&1
  grep -E "MambaSR|mamba dim|VAL |EARLY|Error|Traceback" "$LOG/p12_${NAME}.train.log" | tail -20
  local CK="$OUT/ckpt_best.pt"
  [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (official 100 pairs + MUSIQ/MANIQA)  ckpt=$(basename "$CK")"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/p12_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/p12_${NAME}.eval.log" | tail -3
}

T0=$(date +%s)

B1=$(probe_batch 128 2 "384 320 256 192")
say "M1 batch chosen: $B1"
train_and_eval mamba_c128 "$B1" 1600 --base 128

B2=$(probe_batch 256 2 "192 160 128 96")
say "M2 batch chosen: $B2"
train_and_eval mamba_c256 "$B2" 1000 --base 256

say "SERIAL 12h PLAN DONE  total $(( ($(date +%s) - T0) / 60 )) min"
