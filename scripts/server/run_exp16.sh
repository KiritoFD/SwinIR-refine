#!/usr/bin/env bash
# ~16h EXPERIMENTAL branch campaign (baselines paused). Innovation arms on the
# champion pixel-U-Net recipe (init=b64_ft15k, Muon5e-3, wavelet λ8+shift4):
#   #1 OT loss      : SWD / Sinkhorn  ×  image / wavelet  ×  add / replace-L1   (6)
#   #perceptual     : maximize cached MUSIQ / MANIQA as loss + high-λ sharpen   (6)
#   #4 LoRA         : freeze base, train low-rank adapter (r=8/16) on champion   (3)
#   #2 DCP          : inference data-consistency back-projection on the best ckpt (eval-only)
# All A-strict eval (non-TTA + TTA + IQA). GPU is single-arm-saturated -> serial.
set -uo pipefail
cd /home/ds/realsr || exit 1
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA='data/RealSR(V3)'
CHAMP="/home/ds/realsr/experiments/diffusion/b64_ft15k/ckpt_best.pt"
BEST="/home/ds/realsr/experiments/diffusion/mech_best/p_shift4/ckpt_best.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG=experiments/diffusion/logs; EXP=experiments/diffusion/experiment
mkdir -p "$LOG" "$EXP/ot" "$EXP/perc" "$EXP/lora" "$EXP/dcp"
say(){ echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu(){ for _ in $(seq 1 480); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M<2000 )) && return 0; sleep 15; done; }
CORE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 --warmup 500 --steps 10000 \
  --ema 0.999 --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5"
BASE="--init $CHAMP --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4"
ev2(){ # ckpt dir
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$2/eval_iqa" > "$2.eval.log" 2>&1
  grep -E "OFFICIAL|FAIL" "$2.eval.log" | tail -1
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$2/eval_iqa_tta" > "$2.tta.log" 2>&1
  grep -E "OFFICIAL|FAIL" "$2.tta.log" | tail -1
}
arm(){ # $1 subdir $2 name, rest flags ; returns ckpt via eval
  local SUB="$1" N="$2"; shift 2; local DIR="$EXP/$SUB/$N"
  [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP $SUB/$N"; return 0; }
  wait_gpu; say "ARM $SUB/$N ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" $CORE $BASE "$@" > "$LOG/ex_${SUB}_${N}.train.log" 2>&1
  grep -E "LoRA|NR-IQA|done|Error|Traceback|nan|out of memory" "$LOG/ex_${SUB}_${N}.train.log" | tail -6
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"; ev2 "$CK" "$DIR/eval_iqa"
}

# #1 optimal transport (6)
arm ot ot_swd_img_add  --ot-loss swd --ot-on image --ot-weight 30
arm ot ot_swd_wav_add  --ot-loss swd --ot-on wavelet --ot-weight 30
arm ot ot_swd_wav_rep  --ot-loss swd --ot-on wavelet --ot-weight 8 --ot-replace-l1
arm ot ot_swd_img_rep  --ot-loss swd --ot-on image --ot-weight 8 --ot-replace-l1
arm ot ot_sink_img_add --ot-loss sinkhorn --ot-on image --ot-weight 0.002
arm ot ot_sink_wav_add --ot-loss sinkhorn --ot-on wavelet --ot-weight 0.002
# perceptual: NR-IQA loss + high-λ (6)
arm perc perc_musiq02 --nr-loss musiq --nr-weight 0.02
arm perc perc_musiq05 --nr-loss musiq --nr-weight 0.05
arm perc perc_maniqa1 --nr-loss maniqa --nr-weight 1.0
arm perc perc_maniqa3 --nr-loss maniqa --nr-weight 3.0
arm perc perc_lam16   --dwt-loss --dwt-weight 16 --dwt-levels 2 --dwt-shift 4
arm perc perc_lam20   --dwt-loss --dwt-weight 20 --dwt-levels 2 --dwt-shift 4
# #4 LoRA adapter on the champion (3)  [no --init override needed: BASE already inits champ; but lora wraps trained net]
lora(){ local N="$1"; shift; local DIR="$EXP/lora/$N"
  [ -f "$DIR/eval_iqa/eval.json" ] && { say "SKIP lora/$N"; return 0; }
  wait_gpu; say "ARM lora/$N ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" --backbone unet --size S --base 64 \
    --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg --residual 1 --lr-patch 64 --batch 128 --amp \
    --num-workers 12 --cache-data 0 --warmup 500 --steps 5000 --ema 0.999 --eval-every 1000 --val-pairs 16 \
    --patience 15 --min-steps 2000 --val-eval-steps 8 --save-every 2500 --optimizer muon --muon-lr 5e-3 \
    --lora-r "$1" --freeze-base --init "$BEST" --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 "${@:2}" \
    > "$LOG/ex_lora_${N}.train.log" 2>&1
  grep -E "LoRA|trainable|done|Error|Traceback|nan|out of memory" "$LOG/ex_lora_${N}.train.log" | tail -6
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"; ev2 "$CK" "$DIR/eval_iqa"
}
lora lora_r8  8
lora lora_r16 16
lora lora_r8_ot 8 --ot-loss swd --ot-on wavelet --ot-weight 20
# #2 DCP: inference-only, on the champion (no training)
say "DCP inference on champion $BEST"
if [ -f "$BEST" ]; then
  for it in 3 8; do
    "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$BEST" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --dcp "$it" --dcp-eta 0.3 \
      --out "$EXP/dcp/dcp${it}/eval_iqa_tta" > "$LOG/ex_dcp_${it}.log" 2>&1
    grep -E "OFFICIAL|FAIL" "$LOG/ex_dcp_${it}.log" | tail -1
  done
fi
say "EXPERIMENT CAMPAIGN DONE"
