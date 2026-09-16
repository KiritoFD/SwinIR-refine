#!/usr/bin/env bash
# Round 3: same harness, but with --num-workers exposed.
# Round 1 showed the pixel arms are dataloader-bound (GPU 0% util, workers at
# 99% CPU on a 24-core box), so batch size alone is not the lever there.
#   usage: bash scripts/server/bench_batch3.sh [steps]
set -uo pipefail

ROOT="${ROOT:-/home/ds/realsr}"
PY="${PY:-/home/ds/miniconda3/envs/harness-qwen/bin/python}"
DATA="${DATA:-$ROOT/data/RealSR(V3)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
STEPS="${1:-200}"
cd "$ROOT" || exit 1

OUT="$ROOT/experiments/diffusion"
mkdir -p "$OUT"

# tag  kind size vae batch patch workers
CONFIGS="
lat_S_64    latent S  flux1-vae 64  256 4
lat_XS_96   latent XS flux1-vae 96  256 4
pix_S_24    pixel  S  -         24  64  16
pix_S_32    pixel  S  -         32  64  16
pix_XS_32   pixel  XS -         32  64  16
pix_XS_64   pixel  XS -         64  64  16
"

printf "%-12s %6s %6s %9s %10s %10s\n" CONFIG BATCH WORK PEAK_GB S_PER_STEP SMP_PER_S
echo "$CONFIGS" | while read -r tag kind size vae batch patch wk; do
  [[ -z "$tag" ]] && continue
  rm -rf "$OUT/_bench_$tag"
  if [[ "$kind" == "latent" ]]; then
    cmd=("$PY" -m diffusion.train_latent --data-root "$DATA" --out "$OUT/_bench_$tag"
         --vae "$vae" --latent-cache "data/latents/$vae" --size "$size"
         --objective flow --steps "$STEPS" --lr-patch "$patch" --batch "$batch"
         --amp --eval-every 100000 --save-every 100000 --val-pairs 4 --num-workers "$wk")
  else
    cmd=("$PY" -m diffusion.train_pixel --data-root "$DATA" --out "$OUT/_bench_$tag"
         --size "$size" --objective flow --steps "$STEPS" --lr-patch "$patch"
         --batch "$batch" --amp --eval-every 100000 --save-every 100000
         --val-pairs 4 --num-workers "$wk")
  fi
  log=$(mktemp)
  "${cmd[@]}" >"$log" 2>&1
  rows=$(grep -E '^step' "$log" | tail -2 |
    sed -n 's/^step 0*\([0-9]*\)\/.*| \([0-9.]*\)GB | \([0-9]*\)s$/\1 \2 \3/p')
  if [[ -z "$rows" ]]; then
    echo "$tag FAILED:"
    grep -iE "error|out of memory|Traceback" "$log" | tail -3 | sed 's/^/    /'
  else
    printf '%s\n' "$rows" | awk -v t="$tag" -v b="$batch" -v w="$wk" '
      NR==1 {n1=$1; g=$2; s1=$3}
      NR==2 {n2=$1; s2=$3}
      END {
        r = (s2-s1)/(n2-n1)
        printf "%-12s %6s %6s %9s %10.3f %10.1f\n", t, b, w, g, r, b/r
      }'
  fi
  rm -f "$log"
  rm -rf "$OUT/_bench_$tag"
done
