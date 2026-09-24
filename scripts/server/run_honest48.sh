#!/usr/bin/env bash
# Honest-list campaign (replaces the 20-arm config sweep).  The GPU is already
# saturated by one b64 stride-1 arm (~100% util, ~390/450W), so wall-clock is
# precious -- spend it on the levers that actually address the identified
# bottleneck (390-pair data diversity), not marginal config nudges.
# Chained: waits for champmuon (the Muon delivery champion) to finish, then:
#   N3  adversarial degradation-mining pretrain (G_phi attacks the L1+wavelet-HF
#       objective = "attack where HF recovery is weakest") -> RealSR finetune(Muon+lambda8).
#       This is the data lever -- the one thing never really tested. Pretrain is
#       AdamW (muon+adv would need a GradScaler, currently off for muon); the FT is
#       Muon 5e-3 so the A/B vs the champion isolates the PRETRAIN source.
#   flag mini-sweep (cheap, on the champion recipe): Muon momentum {0.90,0.98},
#       ns_steps 7, per-level lambda "1.5,0.7"/"0.7,1.5", LL term 0.3.
# Judge by SSIM/MUSIQ/MANIQA (PSNR secondary), +-0.05 = tie.  No grad-ckpt.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
PRE="data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K"
PREVAL="data/pretrain/DIV2K_valid_HR"
BASE_EVAL="$ROOT/experiments/diffusion/stack_dwt/b64_pre_dwt8_muon/eval_iqa/eval.json"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
mkdir -p "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
ev() { # $1 ckpt, $2 out, [$3 tta]
  local EXTRA=""; [ "${3:-}" = tta ] && EXTRA="--tta"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$1" \
    --mode pixel --objective reg --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa $EXTRA \
    --out "$2" > "$LOG/hn_$(basename $(dirname $2)).log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/hn_$(basename $(dirname $2)).log" | tail -2
}

echo "[honest48 $(date +%m-%d\ %H:%M)] waiting for champmuon eval ($BASE_EVAL)..."
while [ ! -f "$BASE_EVAL" ]; do
  tmux has-session -t champmuon 2>/dev/null || { [ -f "$BASE_EVAL" ] && break; sleep 30; }
  [ -f "$BASE_EVAL" ] && break; sleep 30
done
wait_gpu
echo "[honest48 $(date +%H:%M)] base ready"

# ---------- N3: adversarial-degradation pretrain -> Muon finetune ----------
ADV="$ROOT/experiments/diffusion/n3/adv_pretrain_v2"; FT="$ROOT/experiments/diffusion/n3/n3_ft"
mkdir -p "$ROOT/experiments/diffusion/n3"
if [ ! -f "$ADV/ckpt_best.pt" ] && [ ! -f "$ADV/ckpt_last.pt" ]; then
  wait_gpu
  say "N3 pretrain: adversarial min-max on L1+wavelet-HF (AdamW)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$ADV" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 25000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 \
    --adv-deg --pretrain-root "$PRE" --pretrain-val-root "$PREVAL" \
    --dwt-loss --dwt-weight 8 --dwt-levels 2 > "$LOG/hn_adv_pretrain.log" 2>&1
  grep -E "adversary|VAL |done|Error|Traceback|nan" "$LOG/hn_adv_pretrain.log" | tail -15
fi
ACK="$ADV/ckpt_best.pt"; [ -f "$ACK" ] || ACK="$ADV/ckpt_last.pt"
if [ -f "$ACK" ] && [ ! -f "$FT/eval_iqa/eval.json" ]; then
  wait_gpu
  say "N3 finetune (init adv_pretrain_v2, Muon 5e-3 + lambda8)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$FT" \
    --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
    --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
    --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
    --lr 3e-4 --warmup 500 --steps 10000 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 \
    --init "$ACK" --optimizer muon --muon-lr 5e-3 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
    > "$LOG/hn_n3_ft.train.log" 2>&1
  grep -E "optimizer=|init weights|VAL |done|Error|Traceback|nan" "$LOG/hn_n3_ft.train.log" | tail -10
  CK="$FT/ckpt_best.pt"; [ -f "$CK" ] || CK="$FT/ckpt_last.pt"
  ev "$CK" "$FT/eval_iqa/eval.json"; ev "$CK" "$FT/eval_iqa_tta/eval.json" tta
fi

# ---------- honest flag mini-sweep on the champion recipe ----------
RECIPE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 \
  --objective reg --residual 1 --decoded-manifest data/decoded/manifest.json \
  --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 \
  --warmup 500 --steps 10000 --ema 0.999 --init $CHAMP --optimizer muon --muon-lr 5e-3 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500"
SWROOT="$ROOT/experiments/diffusion/honest_flags"
flag() { # $1 name, rest = flags
  local NAME="$1"; shift
  local OUT="$SWROOT/$NAME"
  if [ -f "$OUT/eval_iqa/eval.json" ]; then say "SKIP $NAME"; return 0; fi
  wait_gpu
  say "FLAG $NAME ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$OUT" $RECIPE "$@" > "$LOG/hn_${NAME}.train.log" 2>&1
  grep -E "optimizer=|VAL |done|Error|Traceback|nan" "$LOG/hn_${NAME}.train.log" | tail -8
  local CK="$OUT/ckpt_best.pt"; [ -f "$CK" ] || CK="$OUT/ckpt_last.pt"
  ev "$CK" "$OUT/eval_iqa/eval.json"
}
flag f_mom090 --muon-momentum 0.90
flag f_mom098 --muon-momentum 0.98
flag f_ns7   --muon-ns-steps 7
flag f_lv1507 --dwt-level-weights "1.5,0.7"
flag f_lv0715 --dwt-level-weights "0.7,1.5"
flag f_ll03  --dwt-w-ll 0.3
say "HONEST-48 DONE"
