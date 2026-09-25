#!/usr/bin/env bash
# Muon-first campaign (replaces honest48/honest_deep).  Per the directive: tune
# Muon properly first, then run all mechanism experiments with the best Muon.
#   Phase M : one-factor sweep of Muon's own hyper-parameters (lr / momentum /
#             ns_steps / aux-lr / weight-decay) on the delivery base
#             (init=BSRGAN-pretrained b64_ft15k + wavelet-HF lambda8).  Picks the
#             best by SSIM/MUSIQ/MANIQA and writes muon_best.env.
#   Phase mech: shift-ensemble loss, DTCWT, dual-branch wavelet U-Net, and N3
#             adversarial pretrain -- ALL on Muon(best).  GPU is saturated by one
#             arm so this is inherently serial (~28-32h).  No grad-ckpt.
set -uo pipefail
ROOT=/home/ds/realsr
PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
DATA="$ROOT/data/RealSR(V3)"
CHAMP="$ROOT/experiments/diffusion/b64_ft15k/ckpt_best.pt"
PRE="data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K"
PREVAL="data/pretrain/DIV2K_valid_HR"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT" || exit 1
find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null
LOG="$ROOT/experiments/diffusion/logs"
TUNE="$ROOT/experiments/diffusion/muon_tune"
MECH="$ROOT/experiments/diffusion/muon_mech"
mkdir -p "$TUNE" "$MECH" "$LOG"
say() { echo; echo "==================== $*  [$(date +%m-%d\ %H:%M:%S)] ===================="; }
wait_gpu() { for _ in $(seq 1 240); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); (( M < 2000 )) && return 0; sleep 15; done; }
# common: unet S base64 native-lr0 reg residual dwt lambda8 ; Muon default 5e-3
BASE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --amp --num-workers 12 \
  --cache-data 0 --warmup 500 --steps 10000 --ema 0.999 --optimizer muon --muon-lr 5e-3 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --eval-every 500 --val-pairs 16 --patience 15 \
  --min-steps 3000 --val-eval-steps 8 --save-every 2500"
train_eval() {  # $1 dir  rest = flags (must include --out already set by caller via --out $1)
  local DIR="$1"; shift
  if [ -f "$DIR/eval_iqa/eval.json" ]; then say "SKIP $(basename $DIR)"; return 0; fi
  wait_gpu
  say "TRAIN $(basename $DIR) ($*)"
  "$PY" -u -m diffusion.train_pixel --data-root "$DATA" --out "$DIR" "$@" > "$LOG/mf_$(basename $DIR).train.log" 2>&1
  grep -E "DWT-DUAL|optimizer=|adversary|init weights|VAL |done|Error|Traceback|nan" "$LOG/mf_$(basename $DIR).train.log" | tail -8
  local CK="$DIR/ckpt_best.pt"; [ -f "$CK" ] || CK="$DIR/ckpt_last.pt"
  "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
    --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --out "$DIR/eval_iqa" > "$LOG/mf_$(basename $DIR).eval.log" 2>&1
  grep -E "OFFICIAL|FAIL|Error" "$LOG/mf_$(basename $DIR).eval.log" | tail -2
}

# ---------------- Phase M : Muon hyper-parameter sweep ----------------
say "PHASE M (Muon sweep) -- base reference muon5e-3/mom0.95/ns5 = champmuon (SSIM.9280/MUSIQ55.85/MANIQA.3521)"
for spec in \
  "m_lr2e3  | --muon-lr 2e-3" \
  "m_lr8e3  | --muon-lr 8e-3" \
  "m_lr12e3 | --muon-lr 1.2e-2" \
  "m_mom090 | --muon-momentum 0.90" \
  "m_mom098 | --muon-momentum 0.98" \
  "m_ns3    | --muon-ns-steps 3" \
  "m_ns7    | --muon-ns-steps 7" \
  "m_aux3e4 | --muon-aux-lr 3e-4" \
  "m_wd1e2  | --weight-decay 1e-2" ; do
  NAME="${spec%%|*}"; FLAGS="${spec#*|}"; NAME="$(echo $NAME | tr -d ' ')"
  train_eval "$TUNE/$NAME" $BASE --init "$CHAMP" $FLAGS
done

# ---- pick best Muon by SSIM/MUSIQ/MANIQA, write muon_best.env ----
"$PY" - "$TUNE" "$CHAMP" > "$TUNE/muon_best.env" 2>>"$LOG/mf_selector.err" <<'PY'
import json,sys,glob,os
tune=sys.argv[1]
best=None
rows=[]
for f in sorted(glob.glob(os.path.join(tune,"*","eval_iqa","eval.json"))):
    try: j=json.load(open(f))
    except Exception: continue
    if "ssim_y" not in j: continue
    sc=(j["ssim_y"]-0.925)*1000+(j.get("musiq",55.0)-55.0)+(j.get("maniqa",0.345)-0.345)*1000
    d=os.path.basename(os.path.dirname(os.path.dirname(f)))
    try: a=json.load(open(os.path.join(tune,d,"args.json")))
    except Exception: a={}
    rows.append((sc,d,a))
    if best is None or sc>best[0]: best=(sc,d,a)
# include the un-swept BASE reference (champmuon = muon lr5e-3/mom.95/ns5) as a
# candidate too -- without this the selector only compared the 9 deviations and
# could pick a handicapped lr2e-3 over the actually-best 5e-3.
cp = os.path.join(tune, "..", "stack_dwt", "b64_pre_dwt8_muon", "eval_iqa", "eval.json")
if os.path.exists(cp):
    try:
        j = json.load(open(cp))
        sc = (j["ssim_y"]-0.925)*1000+(j.get("musiq",55.0)-55.0)+(j.get("maniqa",0.345)-0.345)*1000
        a = {"muon_lr":0.005,"muon_momentum":0.95,"muon_ns_steps":5,"muon_aux_lr":0.0,"weight_decay":0.0}
        rows.append((sc,"base_5e3",a))
        if best is None or sc>best[0]: best=(sc,"base_5e3",a)
    except Exception:
        pass
# include the base reference (champmuon) as a candidate too
sys.stderr.write("rank:\n"+"\n".join(f"{s:.3f} {d}" for s,d,_ in sorted(rows,reverse=True))+"\n")
if best:
    _,d,a=best
    lr=a.get("muon_lr"); mom=a.get("muon_momentum"); ns=a.get("muon_ns_steps")
    aux=a.get("muon_aux_lr"); wd=a.get("weight_decay")
    print(f'MUON_LR="{lr}"; MUON_MOM="{mom}"; MUON_NS="{ns}"; MUON_AUX="{aux}"; MUON_WD="{wd}"; MUON_BEST_NAME="{d}"')
    sys.stderr.write(f"BEST={d} {lr} {mom} {ns} {aux} {wd}\n")
else:
    print('MUON_LR="5e-3"; MUON_MOM="0.95"; MUON_NS="5"; MUON_AUX="0.0"; MUON_WD="0.0"; MUON_BEST_NAME="fallback"')
PY
# shellcheck disable=SC1091
source "$TUNE/muon_best.env"
say "BEST MUON: $MUON_BEST_NAME lr=$MUON_LR mom=$MUON_MOM ns=$MUON_NS aux=$MUON_AUX wd=$MUON_WD"
MU="--optimizer muon --muon-lr $MUON_LR --muon-momentum $MUON_MOM --muon-ns-steps $MUON_NS --muon-aux-lr $MUON_AUX --weight-decay $MUON_WD"

# ---------------- Phase mech : all on Muon(best) ----------------
MECBASE="--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
  --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 128 --amp --num-workers 12 \
  --cache-data 0 --warmup 500 --steps 10000 --ema 0.999 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
  --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500"

# pretrain+mech (init champ, Muon best)
train_eval "$MECH/p_shift4"   $MECBASE $MU --init "$CHAMP" --dwt-shift 4
train_eval "$MECH/p_dtcwt"    $MECBASE $MU --init "$CHAMP" --dwt-basis dtcwt
train_eval "$MECH/p_lv3shift" $MECBASE $MU --init "$CHAMP" --dwt-levels 3 --dwt-shift 4
train_eval "$MECH/p_aniso"    $MECBASE $MU --init "$CHAMP" --dwt-w-hl 1.5 --dwt-w-lh 0.8 --dwt-shift 4
# zero-pretrain mech (arch; can't init a dual net from a plain-unet ckpt)
train_eval "$MECH/z_dual"        $MECBASE $MU --dwt-dual
train_eval "$MECH/z_dual_shift4" $MECBASE $MU --dwt-dual --dwt-shift 4
train_eval "$MECH/z_dual_dtcwt"  $MECBASE $MU --dwt-dual --dwt-basis dtcwt --lr-patch 64
# N3 adversarial pretrain on Muon(best) -> ft on Muon(best)
N3="$MECH/n3muon"; ADV="$N3/pretrain"; FT="$N3/ft"
if [ ! -f "$ADV/ckpt_last.pt" ] && [ ! -f "$ADV/ckpt_best.pt" ]; then
  train_eval "$ADV" --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg \
    --residual 1 --decoded-manifest data/decoded/manifest.json --lr-patch 64 --batch 128 --amp --num-workers 12 \
    --cache-data 0 --warmup 500 --steps 25000 --ema 0.999 --dwt-loss --dwt-weight 8 --dwt-levels 2 \
    --eval-every 500 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 \
    $MU --adv-deg --pretrain-root "$PRE" --pretrain-val-root "$PREVAL"
fi
ACK="$ADV/ckpt_best.pt"; [ -f "$ACK" ] || ACK="$ADV/ckpt_last.pt"
if [ -f "$ACK" ]; then
  train_eval "$FT" $MECBASE $MU --init "$ACK"
  # TTA eval of the N3 finetune
  if [ ! -f "$FT/eval_iqa_tta/eval.json" ]; then
    CK="$FT/ckpt_best.pt"; [ -f "$CK" ] || CK="$FT/ckpt_last.pt"
    wait_gpu
    "$PY" -u -m diffusion.eval_official --data-root "$DATA" --ckpt "$CK" --mode pixel --objective reg \
      --tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa --tta --out "$FT/eval_iqa_tta/eval.json" \
      > "$LOG/mf_n3_tta.log" 2>&1; grep -E "OFFICIAL|Error" "$LOG/mf_n3_tta.log" | tail -1
  fi
fi
say "MUON-FIRST CAMPAIGN DONE (best Muon=$MUON_BEST_NAME; results under muon_tune/ and muon_mech/)"
