#!/usr/bin/env bash
# Muon close-out: a matched optimizer A/B on the wavelet champion recipe
# (zero-pretrain b64 + dwt-loss lambda8).  Everything identical except the
# optimiser.  Muon reuses model/optim.Muon (Newton-Schulz on >=2D + AdamW aux),
# at its own lr (2e-3, cosine-scheduled like AdamW's 3e-4).  Judge by
# SSIM/MUSIQ/MANIQA (PSNR secondary), +-0.05 dB = tie.
#   muon_adamw : AdamW 3e-4   } matched pair (AdamW lambda8 zero-pretrain
#   muon_muon  : Muon   2e-3   } baseline wasn't run yet -- this is its anchor)
# Computation-bound: no grad-ckpt.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="$ROOT/experiments/diffusion/muon_ab"
mkdir -p "$OUT_ROOT" "$LOG"
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --steps 10000 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 120); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
run() {  # $1 name, rest = optimizer flags
  local NAME="$1"; shift
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME"; return 0; fi
  wait_gpu
  say "TRAIN $NAME ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" $RECIPE "$@" > "$LOG/mu_${NAME}.train.log" 2>&1
  grep -E "optimizer=|PixelUNet|VAL |EARLY|done|Error|Traceback|nan" "$LOG/mu_${NAME}.train.log" | tail -20
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/mu_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mu_${NAME}.eval.log" | tail -2
}
run muon_adamw --optimizer adamw --lr 3e-4
run muon_muon  --optimizer muon  --muon-lr 2e-3
say "MUON A/B DONE"
