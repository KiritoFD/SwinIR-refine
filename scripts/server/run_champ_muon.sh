#!/usr/bin/env bash
# Payoff of the Muon win: apply Muon (lr 5e-3, the sweep optimum) to the DELIVERY
# recipe -- init from the BSRGAN-pretrained champion b64_ft15k, fine-tune with
# wavelet-HF lambda8 + Muon, then eval non-TTA and with TTA.  Muon alone already
# beat AdamW-pretrain on the 3 perceptual metrics, so Muon+pretrain should stack.
# Judge by SSIM/MUSIQ/MANIQA (PSNR secondary).  No grad-ckpt (computation-bound).
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
OUT_ROOT="$ROOT/experiments/diffusion/stack_dwt"
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --steps 10000 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 \
  --val-eval-steps 8 --save-every 2500"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 120); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }

NAME=b64_pre_dwt8_muon
OUT="$OUT_ROOT/$NAME"
if [ ! -f "$OUT/eval_iqa/eval.json" ]; then
  wait_gpu
  say "TRAIN $NAME (init=$CHAMP, muon lr 5e-3 + dwt lambda8)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" $RECIPE \
    --init "$CHAMP" --optimizer muon --muon-lr 5e-3 > "$LOG/sd_${NAME}.train.log" 2>&1
  grep -E "optimizer=|init weights|VAL |EARLY|done|Error|Traceback|nan" "$LOG/sd_${NAME}.train.log" | tail -15
  CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  say "EVAL $NAME (no TTA)"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa \
    --out "$OUT/eval_iqa" > "$LOG/sd_${NAME}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/sd_${NAME}.eval.log" | tail -2
fi
CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
if [ -f "$CK" ] && [ ! -f "$OUT/eval_iqa_tta/eval.json" ]; then
  wait_gpu
  say "TTA EVAL $NAME"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta \
    --out "$OUT/eval_iqa_tta" > "$LOG/sd_${NAME}.tta.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/sd_${NAME}.tta.log" | tail -2
fi
say "CHAMPION MUON RUN DONE"
