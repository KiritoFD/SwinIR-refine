#!/usr/bin/env bash
# Fetch the OFFICIAL baseline repos through the channel that works on this box
# (codeload tarball + api.github.com; git clone and raw.githubusercontent are blocked),
# then print the authoritative architecture configs so our re-implementations can be
# checked against the published param/FLOP counts before any GPU hour is spent.
cd /home/ds/realsr || exit 1
TP=third_party; mkdir -p "$TP"
P=/home/ds/miniconda3/envs/harness-qwen/bin/python

fetch(){ # dir owner repo
  local D="$1" O="$2" R="$3"
  [ -d "$TP/$D" ] && { echo "$D present"; return 0; }
  for BR in main master; do
    code=$(curl -s -o "$TP/$D.tgz" -m 180 -w '%{http_code}' -L "https://codeload.github.com/$O/$R/tar.gz/refs/heads/$BR")
    if [ "$code" = "200" ]; then
      mkdir -p "$TP/$D" && tar xzf "$TP/$D.tgz" -C "$TP/$D" --strip-components 1 && echo "$D OK ($O/$R@$BR, $(du -sh $TP/$D|cut -f1))" && return 0
    fi
  done
  rm -f "$TP/$D.tgz"; echo "$D FAIL ($O/$R)"; return 1
}

echo "===== repo fetch ====="
fetch HAT      XPixelGroup HAT
fetch RCAN     zc-jhu      RCAN
fetch EDSR_OFF sanghyun-son EDSR-PyTorch
fetch SwinIR   JingyunLiang SwinIR
fetch SRFormer caoyuexing  SRFormer

echo; echo "===== tree of what we got (arch/config files only) ====="
for d in HAT RCAN EDSR_OFF SwinIR SRFormer; do
  echo "--- $d"; find "$TP/$d" -maxdepth 3 -type f \( -name '*.py' -o -name '*.json' -o -name '*.yaml' \) 2>/dev/null | head -12
done

echo; echo "===== EDSR-baseline official config (EDSR-PyTorch) ====="
find "$TP/EDSR_OFF" -name 'edsr*.py' -o -name '*demo*' | head -8
grep -rnE "register_model|n_resblocks|n_feats|res_scale|def forward|pixel_shuffle|Upsample" \
  "$TP/EDSR_OFF/src/model/edsr.py" "$TP/EDSR_OFF/src/data/srconfig.py" 2>/dev/null | head -25

echo; echo "===== RCAN official config ====="
grep -rnE "n_resgroups|n_resblocks|n_feats|reduction|scale|upsample|Upsample|def make_body|n_resblocks=|reduction=" \
  $(find "$TP/RCAN" -name '*.py' | head -12) 2>/dev/null | grep -viE "^\s*#" | head -25

echo; echo "===== SwinIR official classical-SR config ====="
grep -rnE "embed_dim|num_heads|swin_depths|swin_window_size|apex_window_size|resblock_num|resblock_rgb_(in|out)|img_size|patch_size|resize_token|nums_dynamic" \
  $(find "$TP/SwinIR" -maxdepth 2 -name '*.py' | head -8) 2>/dev/null | head -30

echo; echo "===== HAT official config(s) ====="
for f in $(find "$TP/HAT/options" -name '*.json' 2>/dev/null | head -8); do
  echo "--- $(basename $f)"; $P -c "
import json,sys
o=json.load(open('$f')); a=o.get('network_g',o)
def g(k):
    for kk,v in o.items():
        pass
    return None
print({k:a.get(k) for k in ('type','embed_dim','depths','num_heads','window_size','mlp_ratio','input_img_channel','patch_unembed_share','upscale','upscale_factor','dapo','conv_scale','mlp_scale','num_in_chs','num_out_chs')})
tr=o.get('train',{}); print('train:', {k:tr.get(k) for k in ('p_size','batch_size','micro_batch_size','iter_number','start_lr','total_iter')})
" 2>/dev/null || head -30 "$f"
done
