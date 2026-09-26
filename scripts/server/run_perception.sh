#!/usr/bin/env bash
# Phase after baselines: (1) extra classic baselines (SRCNN/VDSR/RDN, from scratch, pure L1)
# and (2) PERCEPTION-tuned variants of our model — maximize cached NR-IQA (MUSIQ/MANIQA)
# as a differentiable loss, plus high-λ wavelet sharpening — trading a little PSNR/SSIM for
# higher no-reference perceptual scores, exactly the "MUSIQ 强、SSIM/PSNR 略差" ask.
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
CHAMP="/home/ds/realsr/experiments/diffusion/b64_ft15k/ckpt_best.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG=experiments/diffusion/logs; BL=experiments/diffusion/baselines; PC=experiments/diffusion/perception
mkdir -p "$LOG" "$BL" "$PC"
say(){ echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu(){ for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M<2000 )) && return 0; sleep 15; done; }
ev(){ # $1 ckpt $2 outdir
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$2/eval_iqa" > "$2/../$(basename $2).eval.log" 2>&1
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$2/eval_iqa_tta" > "$2/../$(basename $2).tta.log" 2>&1
}
CORE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 --warmup 500 --steps 10000 \
  --ema 0.999 --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500"
BASECORE="--size S --scale 2 --objective reg --lr-patch 64 --batch 64 --steps 20000 --lr 3e-4 --warmup 500 \
  --ema 0.999 --amp --num-workers 12 --cache-data 0 --eval-every 1000 --val-pairs 16 --patience 20 --min-steps 4000 \
  --val-eval-steps 8 --save-every 5000"

echo "[perception $(date +%H:%M)] waiting for baserep2 (RCAN/SRResNet/RRDB) to finish..."
while tmux has-session -t baserep2 2>/dev/null; do sleep 30; done
wait_gpu

# ---- 1) extra classic baselines (from scratch, pure L1) ----
for B in srcnn vdsr rdn; do
  DIR="$BL/${B}_x2"; [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP bl/$B"; continue; }
  wait_gpu; say "BASELINE $B (from scratch, L1)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" --backbone "$B" $BASECORE \
    > "$LOG/bl_${B}.train.log" 2>&1
  grep -E "done|Error|Traceback|out of memory" "$LOG/bl_${B}.train.log" | tail -5
  CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"; ev "$CK" "$DIR"
done

# ---- 2) perception-tuned variants of OUR model ----
# base = champion finetune (init b64_ft15k, Muon5e-3, wavelet λ8 levels2 shift4)
pec(){ # $1 name, rest = extra flags
  local N="$1"; shift; local DIR="$PC/$N"
  [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP perc/$N"; return 0; }
  wait_gpu; say "PERC $N ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" $CORE \
    --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 \
    --init "$CHAMP" --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 "$@" \
    > "$LOG/pc_${N}.train.log" 2>&1
  grep -E "NR-IQA|done|Error|Traceback|nan|out of memory" "$LOG/pc_${N}.train.log" | tail -6
  CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"; ev "$CK" "$DIR"
}
pec perc_musiq02 --nr-loss musiq --nr-weight 0.02
pec perc_musiq05 --nr-loss musiq --nr-weight 0.05
pec perc_maniqa1 --nr-loss maniqa --nr-weight 1.0
pec perc_maniqa3 --nr-loss maniqa --nr-weight 3.0
pec perc_lam16   --dwt-weight 16
pec perc_lam20   --dwt-weight 20
say "PERCEPTION+EXTRA-BASELINES DONE"
