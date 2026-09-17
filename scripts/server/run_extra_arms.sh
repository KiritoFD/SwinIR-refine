#!/usr/bin/env bash
# Extra arm discovered mid-run: latent flow on the RESIDUAL.
#
# The matrix had a confound: pixel flow uses --residual (models HR - bicubic,
# default 1) while latent flow had no such option and had to generate the full
# HR latent from noise. The reg arms are residual in BOTH spaces, so the
# headline "cost of sampling = reg - flow" was measured against a handicapped
# flow arm. This arm gives the missing residual-vs-residual cell.
set -uo pipefail

export ROOT="${ROOT:-/home/ds/realsr}"
export PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
export DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

LAT="--lr-patch 256 --batch 64 --amp --num-workers 8 --compile --eval-every 1000 \
     --val-pairs 16 --patience 8 --min-steps 6000 --val-eval-steps 8 --save-every 5000"

ts() { date +"%m-%d %H:%M:%S"; }
say() { echo; echo "==================== $*  [$(ts)] ===================="; }

say "TRAIN latent_flow_res (latent / flow / residual, flux VAE, S)"
"$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/latent_flow_res" \
  --vae flux1-vae --latent-cache data/latents/flux1-vae \
  --size S --objective flow --residual 1 --steps 22000 $LAT \
  2>&1 | tee "$LOG/latent_flow_res.train.log" | grep -E "LatentDiT|compile|VAL |EARLY|Error|Traceback" | tail -40

say "EVAL latent_flow_res"
if [[ -f "$OUT/latent_flow_res/ckpt_best.pt" ]]; then
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/latent_flow_res/ckpt_best.pt" --vae flux1-vae --mode latent \
    --tile 256 --pad 16 --tile-batch 8 --steps 20 \
    --sweep-steps 2,4,8,16,32 --sweep-pairs 8 \
    --out "$OUT/latent_flow_res/eval_official" 2>&1 | tee "$LOG/latent_flow_res.eval.log" \
    | grep -E "OFFICIAL|SWEEP|FAIL" | tail -10
fi

# ---------------------------------------------------------------------------
# latent_reg_flux peaked at VAL 30.47 on step 1000 -- i.e. exactly at the end
# of the LR warmup -- and then decayed to 29.2 for the next 8k steps before
# early-stopping.  Official 31.12 therefore comes from a 1k-step checkpoint,
# which understates what regression can do in latent space.  The flux latent is
# scaled by 0.3611, so gradients in normalised units are ~2.8x larger than in
# pixel space and 1e-4 is simply too hot.  Re-run the same arm at 3e-5.
say "TRAIN latent_reg_flux_lr3e5 (latent / reg, flux VAE, S, lr 3e-5)"
"$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/latent_reg_flux_lr3e5" \
  --vae flux1-vae --latent-cache data/latents/flux1-vae \
  --size S --objective reg --lr 3e-5 --steps 16000 $LAT \
  2>&1 | tee "$LOG/latent_reg_flux_lr3e5.train.log" | grep -E "LatentDiT|compile|VAL |EARLY|Error|Traceback" | tail -40

say "EVAL latent_reg_flux_lr3e5"
if [[ -f "$OUT/latent_reg_flux_lr3e5/ckpt_best.pt" ]]; then
  "$PY" -m diffusion.eval_official --data-root "$DATA" \
    --ckpt "$OUT/latent_reg_flux_lr3e5/ckpt_best.pt" --vae flux1-vae --mode latent \
    --tile 256 --pad 16 --tile-batch 8 --steps 20 \
    --out "$OUT/latent_reg_flux_lr3e5/eval_official" 2>&1 | tee "$LOG/latent_reg_flux_lr3e5.eval.log" \
    | grep -E "OFFICIAL|SWEEP|FAIL" | tail -10
fi

say "EXTRA ARMS DONE [$(ts)]"
