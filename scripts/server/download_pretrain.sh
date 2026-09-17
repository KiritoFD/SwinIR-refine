#!/usr/bin/env bash
# Download SwinIR/BSRGAN pretraining HR data.
#
# We only need HR: the LR side is synthesised on the fly with the BSRGAN
# high-order degradation (that is exactly what SwinIR does), so we do NOT
# download any pre-made LR folders.
#
#   DIV2K    train 800 + valid 100   (official ETH mirror)
#   Flickr2K 2650                    (hf-mirror; original SNU host is dead)
set -uo pipefail

D="${D:-/home/ds/realsr/data/pretrain}"
mkdir -p "$D"
cd "$D" || exit 1

LOG="$D/download.log"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

get() {                       # url dest
  local url=$1 dest=$2
  [[ -s "$dest" ]] && { say "have $(basename "$dest") ($(du -h "$dest" | cut -f1))"; return 0; }
  say "GET $(basename "$dest")"
  curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o "$dest.part" "$url" \
    && mv "$dest.part" "$dest" \
    && say "  ok $(du -h "$dest" | cut -f1)" \
    || { say "  FAILED $url"; return 1; }
}

say "target dir $D"
# run the big ones concurrently
get "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" "$D/DIV2K_train_HR.zip" &
p1=$!
get "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" "$D/DIV2K_valid_HR.zip" &
p2=$!
get "https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip" "$D/Flickr2K.zip" &
p3=$!
wait $p1 $p2 $p3
say "downloads finished"

# ---------------------------------------------------------------- unpack
unzip_here() {                # zip  outdir
  local z=$1 o=$2
  [[ -d "$o" ]] && { say "already unpacked $(basename "$o")"; return 0; }
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$z" -d "$D/$(basename "$o")" && say "unzipped -> $o"
  else
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$D/$(basename "$o")" \
      && say "unzipped(py) -> $o"
  fi
}

unzip_here "$D/DIV2K_train_HR.zip" "$D/DIV2K_train_HR"
unzip_here "$D/DIV2K_valid_HR.zip" "$D/DIV2K_valid_HR"
if [[ -s "$D/Flickr2K.zip" ]]; then
  unzip_here "$D/Flickr2K.zip" "$D/Flickr2K"
fi

say "=== counts ==="
for d in "$D"/DIV2K_train_HR "$D"/DIV2K_valid_HR "$D"/Flickr2K; do
  [[ -d "$d" ]] && echo "$(basename "$d"): $(find "$d" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
done
say "ALL DONE"
