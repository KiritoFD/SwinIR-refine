#!/usr/bin/env bash
# Instantiate the OFFICIAL architectures (vendored code, no re-implementation) and
# verify (a) param counts against the papers, (b) LR-in / HR-out shapes at x2.
# sys.path trick: our repo has a top-level `model/` package that shadows the repos'
# `model` package, so we must strip cwd from sys.path before importing them.
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python

echo "===== HAT model zoo / README param lines ====="
grep -nE "HAT-L|HAT-S|params|\| *[0-9]+\.[0-9]+M" third_party/HAT/README.md | head -20
echo "--- HAT-S SRx2 yml (for the small variant) ---"
grep -E "depths|embed_dim|num_heads|mlp_ratio|window_size|upsampler|gt_size|total_iter" third_party/HAT/options/train/train_HAT-S_SRx2_from_scratch.yml 2>/dev/null | head -12

echo; echo "===== official builder entry points ====="
grep -rn "def net(" third_party/SwinIR/models/network_swinir.py | head -3
grep -rn "^from\|^import" third_party/SwinIR/main_test_swinir.py | grep -i swinir | head -3
ls third_party/EDSR_OFF/src/model/ third_party/RCAN/RCAN_TrainCode/code/model/ 2>/dev/null | head -20

echo; echo "===== instantiate + shape check ====="
$P - > experiments/official_counts.log 2>&1 <<'PY'
import os, sys, types, importlib.util, inspect
import torch, torch.nn as nn

REPO = os.getcwd()
# strip cwd / repo root so the vendored repos' own `model` package wins
sys.path = [p for p in sys.path if p not in ("", ".", REPO, REPO + "/model")]

class _Reg:
    def register(self, name=None):
        def deco(cls): return cls
        return deco
for nm in ("basicsr","basicsr.utils","basicsr.utils.registry","basicsr.archs","basicsr.archs.arch_util"):
    sys.modules.setdefault(nm, types.ModuleType(nm))
sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = _Reg()
from timm.layers import to_2tuple, trunc_normal_
sys.modules["basicsr.archs.arch_util"].to_2tuple = to_2tuple
sys.modules["basicsr.archs.arch_util"].trunc_normal_ = trunc_normal_
sys.modules["basicsr.utils"].scandir = lambda *a, **k: []

def n(m): return sum(p.numel() for p in m.parameters()) / 1e6
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m
def shape_test(m, lr=64, scale=2, ch=3):
    x = torch.rand(1, ch, lr, lr)
    with torch.no_grad():
        y = m(x)
    return tuple(y.shape)

# ---------------- HAT (official hat_arch.py, official SRx2 configs)
hat = load("hat_arch_v", REPO + "/third_party/HAT/hat/archs/hat_arch.py")
common = dict(upscale=2, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30,
              conv_scale=0.01, overlap_ratio=0.5, img_range=1.0, mlp_ratio=2,
              upsampler="pixelshuffle", resi_connection="1conv")
for tag, depths in [("HAT-L (repo yml: 12x6)", [6]*12), ("HAT-L variant (6x6)", [6]*6)]:
    m = hat.HAT(depths=depths, num_heads=[6]*len(depths), **common)
    print("%-24s params=%6.2fM  out%s" % (tag, n(m), shape_test(m)))
m = hat.HAT(depths=[6]*6, num_heads=[6]*6, **{**common, "embed_dim": 120})
print("%-24s params=%6.2fM  out%s" % ("HAT-S (120dim,6x6)", n(m), shape_test(m)))

# ---------------- SwinIR (official network_swinir.py, classical-SR config)
sys.path.insert(0, REPO + "/third_party/SwinIR")
swm = load("swinir_v", REPO + "/third_party/SwinIR/models/network_swinir.py")
print("SwinIR builders:", [x for x in dir(swm) if callable(getattr(swm, x)) and x.lower().startswith("net")])
mk = getattr(swm, "net", None)
if mk is not None:
    m = mk(upscale=2, in_chans=3, img_size=48, window_size=8, img_range=1.0,
           depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
           upsampler="pixelshuffle", resi_connection="1conv")
    print("%-24s params=%6.2fM  out%s" % ("SwinIR-L classical", n(m), shape_test(m)))

# ---------------- RCAN (official ECCV18 code)
sys.path.insert(0, REPO + "/third_party/RCAN/RCAN_TrainCode/code")
def purge_model_pkg():
    for k in list(sys.modules):
        if k == "model" or k.startswith("model."):
            del sys.modules[k]
purge_model_pkg()
try:
    rcan = importlib.import_module("model.rcan")
    class ORCAN:
        n_resgroups=10; n_resblocks=20; n_feats=64; reduction=16; res_scale=0.1
        scale=[2]; n_colors=3; rgb_range=255; conv=nn.Conv2d; n_resgroups_extra=0
    m = rcan.rcan(ORCAN)
    print("%-24s params=%6.2fM  out%s" % ("RCAN G10/R20/F64", n(m), shape_test(m)))
except Exception as e:
    print("RCAN failed:", type(e).__name__, e)

# ---------------- EDSR-baseline (official EDSR-PyTorch)
sys.path.insert(0, REPO + "/third_party/EDSR_OFF/src")
purge_model_pkg()
try:
    edsr = importlib.import_module("model.edsr")
    class OEDSR:
        n_resblocks=16; n_feats=64; res_scale=0.1; scale=[2]; n_colors=3; rgb_range=255
        conv=nn.Conv2d; res_dump=0; n_resgroups=1; n_colors_ = 3
        rgb_mean=(0.4488,0.4371,0.4040); rgb_std=(1.,1.,1.); test_only=False
    m = edsr.edsr(OEDSR)
    print("%-24s params=%6.2fM  out%s" % ("EDSR-baseline x2", n(m), shape_test(m)))
except Exception as e:
    print("EDSR failed:", type(e).__name__, e)
PY
tail -25 experiments/official_counts.log
