#!/usr/bin/env bash
# Innovation #1: Optimal-Transport losses (SWD / Sinkhorn) on the champion recipe —
# distribution matching to sidestep the sub-pixel registration wall; expect MUSIQ/MANIQA
# up at some SSIM/PSNR cost (and maybe break the point-to-point PSNR ceiling). Waits for
# the running baselines/perception chains so we never co-schedule on the saturated GPU.
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
CHAMP="/home/ds/realsr/experiments/diffusion/b64_ft15k/ckpt_best.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG=experiments/diffusion/logs; OT=experiments/diffusion/ot
mkdir -p "$LOG" "$OT"
say(){ echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu(){ for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M<2000 )) && return 0; sleep 15; done; }
CORE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 --warmup 500 --steps 10000 \
  --ema 0.999 --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 --init $CHAMP"
arm(){ # $1 name, rest = flags
  local N="$1"; shift; local DIR="$OT/$N"
  [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP $N"; return 0; }
  wait_gpu; say "OT $N ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" $CORE --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 "$@" \
    > "$LOG/ot_${N}.train.log" 2>&1
  grep -E "done|Error|Traceback|nan|out of memory" "$LOG/ot_${N}.train.log" | tail -5
  CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/ot_${N}.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL" "$LOG/ot_${N}.eval.log" | tail -1
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$DIR/eval_iqa_tta" > "$LOG/ot_${N}.tta.log" 2>&1
  grep -E "OFFICIAL|FAIL" "$LOG/ot_${N}.tta.log" | tail -1
}
echo "[ot $(date +%H:%M)] waiting for perc chain (baselines+perception) to finish..."
while tmux has-session -t perc 2>/dev/null || tmux has-session -t baserep2 2>/dev/null || tmux has-session -t baserep 2>/dev/null; do sleep 30; done
wait_gpu
arm ot_swd_img_add  --ot-loss swd --ot-on image    --ot-weight 30
arm ot_swd_wav_add   --ot-loss swd --ot-on wavelet --ot-weight 30
arm ot_swd_wav_rep   --ot-loss swd --ot-on wavelet --ot-weight 8 --ot-replace-l1
arm ot_swd_img_rep   --ot-loss swd --ot-on image    --ot-weight 8 --ot-replace-l1
arm ot_sink_img_add  --ot-loss sinkhorn --ot-on image --ot-weight 0.002
arm ot_sink_wav_add  --ot-loss sinkhorn --ot-on wavelet --ot-weight 0.002
say "OT CAMPAIGN DONE"
