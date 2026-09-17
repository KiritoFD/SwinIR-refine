#!/usr/bin/env bash
# Cut the two VAE-ablation arms (latent_flow_sd, latent_flow_sdxl) out of the
# running v3 matrix.
#
# Why: the VAE noise-floor measurement already settles them.
#   flux1-vae        Y 45.90   <- ceiling
#   sdxl-vae         Y 36.06
#   sd-vae-ft-ema    Y 34.62
# A latent DiT can never beat its VAE's decode ceiling, and the observed
# train-vs-ceiling loss is >5 dB, so sd/sdxl would land near 22-24 dB -- below
# both the existing flux arm (28.36 / 31.12) and pixel_reg (33.76).  Two runs
# costing ~3.1h cannot change any conclusion; that time is worth more as
# pretraining steps.
#
# run_36h3.sh is live, so it must NOT be edited in place (bash re-reads by byte
# offset).  Instead we stop the matrix session the moment the last arm we still
# want (pixel_reg_XS) has been evaluated.  chain_after2.sh is waiting for the
# session to disappear and will pick up from there.
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
OUT="$ROOT/experiments/diffusion"
LOG="$ROOT/experiments/cut_vae.log"
SESS="${SESS:-realsr36h}"
MARKER="$OUT/pixel_reg_XS/eval_official/eval.json"

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "watchdog up: will stop '$SESS' once pixel_reg_XS is evaluated"

for i in $(seq 1 2880); do          # up to 24h
  if [[ -f "$MARKER" ]]; then
    say "pixel_reg_XS eval found -> stopping matrix before latent_flow_sd"
    break
  fi
  # fallback: we slept through the hand-off and sd is already training
  if [[ -f "$OUT/logs/latent_flow_sd.train.log" ]]; then
    last=$(grep -oE "^step [0-9]+" "$OUT/logs/latent_flow_sd.train.log" | tail -1 | awk '{print $2}')
    if [[ -n "${last:-}" ]] && (( 10#${last:-0} >= 200 )); then
      say "missed the marker; latent_flow_sd already at step $last -> stopping now"
      break
    fi
  fi
  tmux has-session -t "$SESS" 2>/dev/null || { say "matrix session ended on its own"; exit 0; }
  sleep 30
done

tmux kill-session -t "$SESS" 2>/dev/null
sleep 5
# belt and braces: make sure no sd/sdxl trainer survives the session teardown.
# The bracket keeps pkill from matching this script's own command line.
pkill -f "latent_flow_s[d]" 2>/dev/null
pkill -f "latent_flow_sdx[l]" 2>/dev/null
sleep 5

if tmux has-session -t "$SESS" 2>/dev/null; then
  say "WARNING: session $SESS still alive"
else
  say "matrix stopped; chain_after2 should pick up stage A"
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
