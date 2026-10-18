#!/usr/bin/env bash
# Pre-flight for FAITHFUL baseline reproduction:
#  (1) can we git-clone the official repos on this box? (github returned 200 earlier)
#  (2) what training data actually exists here (RealSR only? DIV2K? DF2K?)
#  (3) print each official repo's model config so we can match published param counts
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
TP=third_party; mkdir -p "$TP"

echo "===== (1) clone official repos (shallow) ====="
declare -A REPOS=(
  [HAT]="https://github.com/XPixelGroup/HAT.git"
  [SRFormer]="https://github.com/caoyuexing/SRFormer.git"
  [RLFN]="https://github.com/hzy117/RLFN.git"
  [ATD]="https://github.com/LSYAAAAAAA/AdaptiveTokenDictionary.git"
  [RDN]="https://github.com/jixiaozhong/RDN.git"
  [SwinIR]="https://github.com/JingyunLiang/SwinIR.git"
  [EDSR]="https://github.com/joonkyumoon/EDSR-PyTorch.git"
  [EDSR2]="https://github.com/sanghyun-son/EDSR-PyTorch.git"
  [RCAN]="https://github.com/jungbee-pad/RCAN.git"
)
for k in "${!REPOS[@]}"; do
  if [ -d "$TP/$k/.git" ]; then echo "$k already cloned"; continue; fi
  timeout 120 git clone -q --depth 1 "${REPOS[$k]}" "$TP/$k" 2>"$TP/_err_$k" \
    && echo "$k CLONE OK  $(du -sh $TP/$k | cut -f1)" \
    || echo "$k CLONE FAIL: $(head -c 160 $TP/_err_$k | tr '\n' ' ')"
done

echo; echo "===== (2) data available on this box ====="
ls -1 data 2>/dev/null
for d in data/DIV2K data/DF2K data/Flickr2K data/data; do
  [ -e "$d" ] && echo "present: $d ($(find $d -maxdepth 2 -type d | wc -l) dirs, $(find $d -name '*.png' -o -name '*.jpg' | wc -l) imgs)"
done

echo; echo "===== (3) official configs found in the clones ====="
for f in $(ls $TP/HAT/options/*.json 2>/dev/null | head -20); do echo "--- $f"; grep -oE '"(embed_dim|num_heads|depths|dw_scale|window_size|mlp_type|input_size|pretrain_iter)": *[^,}]*' $f | tr '\n' ' '; echo; done
for f in $(ls $TP/SRFormer/configs/*.yaml $TP/SRFormer/options/*.yaml 2>/dev/null | head -20); do echo "--- $f"; grep -iE 'embed_dim|num_heads|depths|window|mlp|network' $f | head -12; done
echo "--- RLFN model def / train opts"
grep -rnE "n_feat|n_res|n_rdb|n_rdbnum|n_rdbnr|res_scale|n_colors" $TP/RLFN/train.py $TP/RLFN/model.py 2>/dev/null | head -20
echo "--- RDN opts"
grep -rnE "nblocks|nLayers|df_dim|growthRate|nBlocks|nDenselayer|nFE" $TP/RDN/train.py $TP/RDN/test.py 2>/dev/null | head -15
echo "--- ATD opts"
grep -rnE "embed_dim|depths|num_heads|token_dict|window|mlp" $TP/ATD/basicsr/basicsr/archs/atd_arch.py 2>/dev/null | head -10
ls $TP/ATD 2>/dev/null | head
