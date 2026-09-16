#!/usr/bin/env bash
# Measure peak GB + s/step for candidate batch sizes, so the 36h plan can be
# sized against the real 48G card instead of extrapolated.
#   usage: bash scripts/server/bench_batch.sh [steps]
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
STEPS="${1:-220}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion"
mkdir -p "$OUT"

# tag  kind size vae batch patch
CONFIGS="
lat_S_64    latent S  flux1-vae 64  256
lat_S_80    latent S  flux1-vae 80  256
lat_XS_96   latent XS flux1-vae 96  256
pix_S_24    pixel  S  -         24  64
pix_XS_32   pixel  XS -         32  64
"

printf "%-12s %8s %10s %10s\n" CONFIG BATCH PEAK_GB S_PER_STEP
echo "$CONFIGS" | while read -r tag kind size vae batch patch; do
  [[ -z "$tag" ]] && continue
  rm -rf "$OUT/_bench_$tag"
  if [[ "$kind" == "latent" ]]; then
    cmd=("$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/_bench_$tag"
         --vae "$vae" --latent-cache "data/latents/$vae" --size "$size"
         --objective flow --steps "$STEPS" --lr-patch "$patch" --batch "$batch"
         --amp --eval-every 100000 --save-every 100000 --val-pairs 4)
  else
    cmd=("$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/_bench_$tag"
         --size "$size" --objective flow --steps "$STEPS" --lr-patch "$patch"
         --batch "$batch" --amp --eval-every 100000 --save-every 100000 --val-pairs 4)
  fi
  log=$(mktemp)
  "${cmd[@]}" >"$log" 2>&1
  # steady-state rate from the last two logged intervals (excludes startup/warmup)
  rows=$(grep -E '^step' "$log" | tail -2 |
    sed -n 's/^step 0*\([0-9]*\)\/.*| \([0-9.]*\)GB | \([0-9]*\)s$/\1 \2 \3/p')
  if [[ -z "$rows" ]]; then
    echo "$tag FAILED:"
    grep -iE "error|out of memory|Traceback" "$log" | tail -3 | sed 's/^/    /'
  else
    printf '%s\n' "$rows" | awk -v t="$tag" -v b="$batch" '
      NR==1 {n1=$1; g=$2; s1=$3}
      NR==2 {n2=$1; s2=$3}
      END {
        if (NR==1) { printf "%-12s %8s %10s %10s\n", t, b, g, "?" }
        else { printf "%-12s %8s %10s %10.3f\n", t, b, g, (s2-s1)/(n2-n1) }
      }'
  fi
  rm -f "$log"
  rm -rf "$OUT/_bench_$tag"
done
