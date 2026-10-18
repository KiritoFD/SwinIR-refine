"""Vendor the OFFICIAL SR baseline architectures from `third_party/` (fetched from each
project's own repo) instead of re-implementing them, so a "reproduction" really is the
paper's model: same body, same upsampling (post-upsampling sub-pixel from the LR input),
same channel/depth/head/window numbers, and a param count we can check against the paper.

Why this file exists: the earlier `baselines_sr.py` arms were *pre-upsampling residual
variants* of these families (they take the bicubic-upsampled image, matching our own U-Net
harness). That was a different model: RCAN came out 3.76M instead of 16.1-16.7M, RDN 0.05M
instead of ~1.3M. Those numbers stay useful as "same-protocol variant" rows but they are
NOT the official baselines, so we do not call them that.

Faithfulness contract, stated exactly:
  * architecture / upsampling / params  : OFFICIAL (loaded from the official source file)
  * training data                       : RealSR Train 390 pairs (user's decision), NOT DF2K
  * loss / optimiser / budget           : one shared recipe for every arm so the ladder
                                         measures the architecture, not the recipe
  * evaluation                          : our A-strict protocol (limited-range Y, uint8,
                                         modcrop4, no shave, 100 pairs, tile 64 / pad 16)
So every number we produce for these arms is "official model, re-trained under our
protocol" - never claimed as the paper's own DIV2K-trained number.

Also note: these nets consume the **LR** image and emit HR (`net_input="lr"` in
train_pixel / eval_official), unlike our U-Net which refines a bicubic upsample.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"

EDSR_SRC = TP / "EDSR_OFF" / "src"                     # sanghyun-son/EDSR-PyTorch (official EDSR code)
RCAN_SRC = TP / "RCAN" / "RCAN_TrainCode" / "code"     # yulunzhang/RCAN (ECCV'18 official)
SWINIR_SRC = TP / "SwinIR"                             # JingyunLiang/SwinIR (official)
HAT_SRC = TP / "HAT"                                   # XPixelGroup/HAT (official)
SRFORMER_SRC = TP / "SRFormer"                         # HVision-NKU/SRFormer (official)
SRFORMER_ARCH = SRFORMER_SRC / "basicsr" / "archs" / "srformer_arch.py"


# --------------------------------------------------------------------- import plumbing
def _install_basicsr_shim() -> None:
    """HAT's arch file imports 3 symbols from basicsr. Provide them without installing
    basicsr (its setup needs a compile step and would drag a different torch in)."""
    class _Reg:
        def register(self, name=None):
            def deco(cls):
                return cls
            return deco

    names = ("basicsr", "basicsr.utils", "basicsr.utils.registry",
             "basicsr.archs", "basicsr.archs.arch_util")
    for nm in names:
        sys.modules.setdefault(nm, types.ModuleType(nm))
    sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = _Reg()
    sys.modules["basicsr.utils"].scandir = lambda *a, **k: []
    try:
        from timm.layers import to_2tuple, trunc_normal_
    except Exception:                                            # pragma: no cover
        from timm.models.layers import to_2tuple, trunc_normal_  # type: ignore
    sys.modules["basicsr.archs.arch_util"].to_2tuple = to_2tuple
    sys.modules["basicsr.archs.arch_util"].trunc_normal_ = trunc_normal_


def _load(modname: str, path: Path, src_root: Path) -> types.ModuleType:
    """Import one official file in isolation.

    Both EDSR-PyTorch and RCAN do `from model import common`, and our repo owns a
    top-level `model/` package - so the target repo's src dir must be first on sys.path
    and any cached `model*` modules must be purged before/after, otherwise we silently
    bind the wrong `model.common`.
    """
    saved_path = list(sys.path)
    str_root = str(src_root)
    sys.path = [p for p in sys.path if p not in ("", ".", str(ROOT))]
    sys.path.insert(0, str_root)
    cached = {k: sys.modules.pop(k) for k in list(sys.modules)
              if k == "model" or k.startswith("model.")}
    try:
        spec = importlib.util.spec_from_file_location(modname, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        m = importlib.util.module_from_spec(spec)
        sys.modules[modname] = m
        spec.loader.exec_module(m)
    finally:
        for k in [k for k in list(sys.modules) if k == "model" or k.startswith("model.")]:
            sys.modules.pop(k, None)
        sys.modules.update(cached)
        sys.path = saved_path
    return m


def _load_pkg(modname: str, path: Path, pkg_dir: Path) -> types.ModuleType:
    """Load one file that uses sibling relative imports (`from .arch_util import ...`).

    Registers a synthetic package whose __path__ is the archs directory, and ALSO stubs
    that sibling `arch_util` module: the repos' real arch_util pulls in `basicsr.ops.*`
    (deformable-conv CUDA extensions we do not have), while the arch file itself only
    needs to_2tuple / trunc_normal_. Stubbing it keeps the official architecture file
    verbatim and avoids dragging in compiled ops that are irrelevant to the model.
    """
    _install_basicsr_shim()
    pkg = modname.rsplit(".", 1)[0]
    if pkg not in sys.modules or getattr(sys.modules[pkg], "__file__", None) is None:
        p = types.ModuleType(pkg)
        p.__path__ = [str(pkg_dir)]
        sys.modules[pkg] = p
    util = pkg + ".arch_util"
    if util not in sys.modules:
        from timm.layers import to_2tuple as _t2, trunc_normal_ as _tn
        u = types.ModuleType(util)
        u.to_2tuple = _t2
        u.trunc_normal_ = _tn
        sys.modules[util] = u
    return _load(modname, path, pkg_dir)


class OfficialNet(nn.Module):
    """Adapt an official net to this harness: forward(x, t) signature, an `align`
    attribute for the tiler, the internal attention `win` (window attention requires
    tile sizes to be multiples of it), and the LR-in / HR-out convention."""

    def __init__(self, net: nn.Module, align: int, name: str, win: int = 0, net_input: str = "lr"):
        super().__init__()
        self.net = net
        self.align = int(align)
        self.win = int(win)
        self.net_input = net_input          # "lr" = upsamples internally, "hr" = refines bicubic
        self.coord_channels = 0
        self.official_name = name

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


class _Args:
    """Minimal stand-in for the repos' argparse `args` object."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------------- the 4 arms
def build_edsr_official(scale: int = 2) -> OfficialNet:
    """EDSR-baseline (Lim et al., CVPR'17 workshop): 16 resblocks x 64 feats, res_scale
    0.1, sub-pixel up, no BN. Paper: 1.52M params."""
    mod = _load("edsr_official", EDSR_SRC / "model" / "edsr.py", EDSR_SRC)
    args = _Args(n_resblocks=16, n_feats=64, res_scale=0.1, scale=[int(scale)], n_colors=3,
                 rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
                 conv=nn.Conv2d, res_dump=0, n_resgroups=1)
    return OfficialNet(mod.EDSR(args), align=1, name="EDSR-baseline(official)")


def build_rcan_official(scale: int = 2) -> OfficialNet:
    """RCAN (Zhang et al., ECCV'18): 10 residual groups x 20 RCAB, 64 feats, r=16,
    sub-pixel up. Paper reports 16.70M for the shared 2/3/4-head model; a single x2 head
    is what we train (printed at build time so the table can state it)."""
    mod = _load("rcan_official", RCAN_SRC / "model" / "rcan.py", RCAN_SRC)
    args = _Args(n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, res_scale=0.1,
                 scale=[int(scale)], n_colors=3, rgb_range=1,
                 rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
                 conv=nn.Conv2d, n_resgroups_extra=0)
    return OfficialNet(mod.RCAN(args), align=1, name="RCAN(official)")


def build_swinir_official(scale: int = 2) -> OfficialNet:
    """SwinIR (Liang et al., ICCV-W'21) *classical SR* config, verbatim from
    main_test_swinir.py: depths [6]*6, embed_dim 180, heads [6]*6, window 8, mlp 2,
    pixelshuffle + 1conv, 4 RSTB pre-conv blocks. Paper: 11.88M params."""
    mod = _load("swinir_official", SWINIR_SRC / "models" / "network_swinir.py", SWINIR_SRC)
    net = mod.SwinIR(upscale=int(scale), in_chans=3, img_size=48, window_size=8, img_range=1.0,
                     depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                     upsampler="pixelshuffle", resi_connection="1conv",
                     num_in_ch=3, num_out_ch=3)
    return OfficialNet(net, align=1, name="SwinIR-L classical(official)", win=8)


def build_hat_official(scale: int = 2) -> OfficialNet:
    """HAT-L (Chen et al., CVPR'23) exactly as options/train/train_HAT-L_SRx2_*.yml:
    12 RHAG x 6 HAB (= "double the depth of HAT", paper sec. 3.4), embed 180, heads 6,
    window 16, mlp 2, compress_ratio 3 (CAB), conv_scale 0.01, overlap_ratio 0.5 (OCA),
    pixelshuffle + 1conv -> 40.70M params (HAT, non-L, is the 20.83M model)."""
    _install_basicsr_shim()
    mod = _load("hat_official", HAT_SRC / "hat" / "archs" / "hat_arch.py", HAT_SRC)
    net = mod.HAT(upscale=int(scale), in_chans=3, img_size=64, window_size=16,
                  compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                  img_range=1.0, depths=[6] * 12, embed_dim=180, num_heads=[6] * 12,
                  mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    return OfficialNet(net, align=1, name="HAT-L(official)", win=16)


def build_vdsr_official(scale: int = 2) -> OfficialNet:
    """VDSR (Kim et al., CVPR'16) with the official deep-supervision-free body:
    18 conv layers + global residual, 64 feats, mean-shift around the net.
    Repo: EDSR-PyTorch model/vdsr.py (the authors' reference implementation)."""
    mod = _load("vdsr_official", EDSR_SRC / "model" / "vdsr.py", EDSR_SRC)
    n_res, n_f = 18, 64                                            # VDSR: 20 conv layers, C=64
    mod.url.setdefault("r%d f%d" % (n_res, n_f), "")               # pretrained-URL table lookup only
    mod.url.setdefault("r{}f{}".format(n_res, n_f), "")
    args = _Args(n_resblocks=n_res, n_feats=n_f, res_scale=1.0, scale=[int(scale)], n_colors=3,
                 rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
                 conv=nn.Conv2d)
    return OfficialNet(mod.VDSR(args), align=1, name="VDSR-18(official)", net_input="hr")


def build_rdn_official(scale: int = 2, config: str = "B", g0: int = 64) -> OfficialNet:
    """RDN (Wang et al., CVPR'18) from the reference implementation (EDSR-PyTorch
    model/rdn.py). Its config table: 'A' = (D=20,C=6,G=32), 'B' = (D=16,C=8,G=64);
    repo default is B with G0=64, kSize=3 - which is what we build (params printed and
    reported, so the row can be matched against the paper's own table)."""
    mod = _load("rdn_official", EDSR_SRC / "model" / "rdn.py", EDSR_SRC)
    args = _Args(scale=[int(scale)], G0=int(g0), RDNkSize=3, RDNconfig=config, n_colors=3)
    return OfficialNet(mod.RDN(args), align=1, name=f"RDN-{config}(official)")


def build_srformer_official(scale: int = 2) -> OfficialNet:
    """SRFormer (Cao et al., ICCV'23) - permuted self-attention. Config verbatim from
    options/train/SRFormer/train_SRFormer_SRx2_scratch.yml: img_size 48, window 24,
    depths [6]*6, embed_dim 180, heads [6]*6, mlp 2, pixelshuffle, 1conv."""
    mod = _load_pkg("sfo.arch.srformer", SRFORMER_ARCH, SRFORMER_ARCH.parent)
    net = mod.SRFormer(upscale=int(scale), in_chans=3, img_size=48, window_size=24,
                       img_range=1.0, depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                       mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    return OfficialNet(net, align=1, name="SRFormer(official)", win=24)


OFFICIAL = {
    "edsr_off": build_edsr_official,
    "rcan_off": build_rcan_official,
    "swinir_off": build_swinir_official,
    "hat_off": build_hat_official,
    "vdsr_off": build_vdsr_official,
    "rdn_off": build_rdn_official,
    "srformer_off": build_srformer_official,
}


def build(name: str, scale: int = 2, **kw) -> OfficialNet:
    if name not in OFFICIAL:
        raise KeyError(f"unknown official baseline {name!r}; have {sorted(OFFICIAL)}")
    return OFFICIAL[name](scale=scale, **kw)


def is_official(name: str) -> bool:
    return name in OFFICIAL


if __name__ == "__main__":
    # quick self-check: python -m diffusion.baselines_official
    for k in OFFICIAL:
        m = build(k, scale=2)
        p = sum(q.numel() for q in m.parameters()) / 1e6
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = m(x, None)
        print(f"{k:11s} {m.official_name:30s} params={p:6.2f}M  64x64 LR -> {tuple(y.shape)}")
