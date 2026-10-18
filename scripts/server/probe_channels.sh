#!/usr/bin/env bash
cd /home/ds/realsr || exit 1
echo "===== what is data/pretrain ? ====="
du -sh data/pretrain 2>/dev/null
find data/pretrain -maxdepth 2 -type d 2>/dev/null | head -12
echo "img count: $(find data/pretrain -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)"
echo "--- sizes (are they DIV2K-scale?) ---"
find data/pretrain -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | head -3 | while read -r f; do
  /home/ds/miniconda3/envs/harness-qwen/bin/python -c "from PIL import Image;import sys;im=Image.open('$f');print('$f'.split('/')[-1], im.size)"
done

echo; echo "===== which fetch channel works ====="
echo "-- raw.githubusercontent:"; curl -s -o /dev/null -m 12 -w '%{http_code}\n' https://raw.githubusercontent.com/XPixelGroup/HAT/main/README.md
echo "-- api.github.com:";       curl -s -o /dev/null -m 12 -w '%{http_code}\n' https://api.github.com/repos/XPixelGroup/HAT
echo "-- codeload tarball:";      curl -s -o /dev/null -m 20 -w '%{http_code} size=%{size_download}\n' -L https://codeload.github.com/XPixelGroup/HAT/tar.gz/refs/heads/main
echo "-- pypi (pip download basicsr):"
/home/ds/miniconda3/envs/harness-qwen/bin/pip download basicsr --no-deps -d /tmp/bsr -q 2>&1 | tail -2
ls -la /tmp/bsr 2>/dev/null | tail -2

echo; echo "===== successful clones: what archs do they carry ====="
ls third_party/EDSR2/master/src/arch/ 2>/dev/null || find third_party/EDSR2 -name '*.py' -path '*arch*' | head -20
echo "--- EDSR-baseline config in the official repo ---"
grep -rnE "edsr.?baseline|n_resblocks|n_feats|res_scale" third_party/EDSR2/master/src/data/demo.py third_party/EDSR2/master/src/train.py third_party/EDSR2/master/src/model/edsr.py 2>/dev/null | head -18
echo "--- vdsr arch present? ---"; ls third_party/EDSR2/master/src/arch/ 2>/dev/null | grep -iE "vdsr|rcan|rdn"
echo "--- SwinIR official configs ---"
grep -rhoE "\-\-(embed_dim|resblock_num|swin_batch|test_patch_size|resize_crop|pretrain_iter) [0-9]+" third_party/SwinIR/main_train_swin.py third_party/SwinIR/*.sh 2>/dev/null | sort -u | head -20
grep -nE "SwinIR_light|SwinIR_base|SwinIR_large|swinir_config" -A6 third_party/SwinIR/networks/network_swinir.py 2>/dev/null | head -30
