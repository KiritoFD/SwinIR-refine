#!/usr/bin/env bash
# Chained after mech_best: TTA-eval the two shift winners (p_shift4 / p_aniso) so the
# delivery champion (pretrain + Muon5e-3 + wavelet-lambda8 + shift-ensemble) gets its
# TTA number too.  TTA-only (no retrain), few minutes each.  Waits for mechbest to end
# so it doesn't fight the N3 pretrain for the (saturated) GPU.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
LOG="$ROOT/experiments/diffusion/logs"; MB="$ROOT/experiments/diffusion/mech_best"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
echo "[tta_best $(date +%H:%M)] waiting for mechbest..."
while tmux has-session -t mechbest 2>/dev/null; do sleep 30; done
wait_gpu
for n in p_shift4 p_aniso; do
  CK="$MB/$n/ckpt_best.pt"; [ -f "$CK" ] || CK="$MB/$n/ckpt_last.pt"
  [ -f "$MB/$n/eval_iqa_tta/eval.json" ] && { say "SKIP $n TTA"; continue; }
  say "TTA EVAL $n"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$MB/$n/eval_iqa_tta" \
    > "$LOG/tb_${n}_tta.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/tb_${n}_tta.log" | tail -2
done
say "TTA-BEST DONE"
