#!/usr/bin/env bash
# Self-driving follow-up to the Muon A/B.  Waits for muon_muon (lr 2e-3) to finish,
# then sweeps Muon's OWN learning rate the way FFN taught us not to judge a lever on a
# bad config: muon is lr-sensitive, so a single 2e-3 loss says nothing.  Runs
# lr = 1e-3 / 5e-3 / 1e-2 on the identical zero-pretrain b64+dwt-lambda8 recipe
# (the already-running 2e-3 gives the 4th point).  Then best-Muon vs AdamW(3e-4)
# decides whether Muon earns a place in the recipe or is a fair neutral.
# Judge by SSIM/MUSIQ/MANIQA (PSNR secondary), +-0.05 = tie.  No grad-ckpt (computation-bound).
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
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --steps 10000 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 120); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

echo "[muon-sweep $(date +%H:%M:%S)] waiting for muon_muon (lr 2e-3) eval..."
while [ ! -f "$OUT_ROOT/muon_muon/eval_iqa/eval.json" ]; do
  # abort the wait only if the muonab driver died WITHOUT producing it
  tmux has-session -t muonab 2>/dev/null || { [ -f "$OUT_ROOT/muon_muon/eval_iqa/eval.json" ] && break; sleep 30; tmux has-session -t muonab 2>/dev/null || { echo "muonab gone, muon_muon missing -> proceeding with sweep anyway"; break; }; }
  sleep 30
done

run_muon() {  # $1 name, $2 lr
  local NAME="$1" LR="$2"
  local OUT="$OUT_ROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME"; return 0; fi
  wait_gpu
  say "TRAIN $NAME (muon lr=$LR)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" $RECIPE \
    --optimizer muon --muon-lr "$LR" > "$LOG/mu_${NAME}.train.log" 2>&1
  grep -E "optimizer=|VAL |EARLY|done|Error|Traceback|nan" "$LOG/mu_${NAME}.train.log" | tail -12
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/mu_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mu_${NAME}.eval.log" | tail -2
}
run_muon muon_lr1e3  1e-3
run_muon muon_lr5e3  5e-3
run_muon muon_lr1e2  1e-2
say "MUON LR SWEEP DONE (compare muon_adamw + muon_{lr1e3,2e3=muon_muon,5e3,1e2})"
