"""Classic SR baselines (EDSR / RCAN), re-implemented for THIS harness so we can put
them on the same A-strict protocol as our U-Net instead of quoting paper numbers from a
different protocol.

Faithfulness notes (be explicit — do not overclaim):
  * This harness feeds the **bicubic-upsampled LR at HR resolution** and adds a residual.
    So we implement the **pre-upsampling** variants (EDSR/RCAN paper §4 ablation: input is
    upsampled first, the network refines at HR) rather than the post-upsampling sub-pixel
    head — the sub-pixel head would upsample a second time and is meaningless on HR input.
  * Bodies are standard: EDSR = residual blocks (conv-ReLU-conv, no norm); RCAN = residual
    groups of channel-attention blocks (RCAB) + CA.  Both return `y - x` so the harness's
    `pred = bicubic(x) + model(x)` reconstructs the refined HR image y.
  * So a reader should treat these as "EDSR-/RCAN-style pre-upsampling baselines under an
    identical protocol", not bit-exact copies of the original repos.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(cin, cout, k, bias=True):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias)


class ResBlock(nn.Module):
    """EDSR residual block (no BN)."""

    def __init__(self, nf, k=3, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(default_conv(nf, nf, k), nn.ReLU(True), default_conv(nf, nf, k))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=16,
                 scale=2, res_scale=0.1, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = default_conv(self.in_channels, nf, 3)
        self.body = nn.Sequential(*[ResBlock(nf, 3, res_scale) for _ in range(n_res)])
        self.tail = default_conv(nf, self.out_channels, 3)
        self.align = 1  # pre-upsampling: no internal down/upsample, any HR size works

    def forward(self, x, t=None):
        y = self.tail(self.body(self.head(x)))
        return y  # harness: pred = bicubic(x) + this


class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nf, max(nf // reduction, 4), 1), nn.ReLU(True),
            nn.Conv2d(max(nf // reduction, 4), nf, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv_du(self.avg(x))


class RCAB(nn.Module):
    def __init__(self, nf, k=3, reduction=16, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(nf, nf, k), nn.ReLU(True), default_conv(nf, nf, k), CALayer(nf, reduction)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class RCAN(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=10,
                 n_groups=5, scale=2, reduction=16, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = default_conv(self.in_channels, nf, 3)
        self.body = nn.Sequential(*[
            nn.Sequential(*[RCAB(nf, 3, reduction) for _ in range(n_res)])
            for _ in range(n_groups)
        ])
        self.body_tail = default_conv(nf, nf, 3)
        self.tail = default_conv(nf, self.out_channels, 3)
        self.align = 1

    def forward(self, x, t=None):
        f = self.head(x)
        f = self.body_tail(self.body(f)) + f
        return self.tail(f)  # harness: pred = bicubic(x) + this


def build_edsr(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return EDSR(**kw)


def build_rcan(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return RCAN(**kw)


# --------------------------------------------------------------------- SRResNet


class SRResNetBlock(nn.Module):
    """SRResNet residual block: conv-BN-ReLU-conv-BN (BN is its signature vs EDSR)."""

    def __init__(self, nf, k=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, k, padding=k // 2), nn.BatchNorm2d(nf), nn.PReLU(),
            nn.Conv2d(nf, nf, k, padding=k // 2), nn.BatchNorm2d(nf),
        )

    def forward(self, x):
        return x + self.body(x)


class SRResNet(nn.Module):
    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=16,
                 scale=2, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = nn.Conv2d(self.in_channels, nf, 9, padding=4)
        self.body = nn.Sequential(*[SRResNetBlock(nf) for _ in range(n_res)])
        self.body_tail = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))
        self.tail = nn.Conv2d(nf, self.out_channels, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        f = self.body(self.head(x))
        f = self.body_tail(f) + self.head(x)
        return self.tail(f)  # pre-upsampling residual form (see module docstring)


def build_srresnet(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return SRResNet(**kw)


# ---------------------------------------------------------------------- RRDB


class DenseBlock(nn.Module):
    """ESRGAN/Real-ESRGAN dense block: 5 convs, each seeing all previous features,
    residual-scaled by beta."""

    def __init__(self, nf, gc=32, k=3, beta=0.2):
        super().__init__()
        self.beta = beta
        self.convs = nn.ModuleList([
            nn.Conv2d(nf + i * gc, gc, k, padding=k // 2) for i in range(5)
        ])
        self.lff = nn.Conv2d(nf + 5 * gc, nf, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feats = [x]
        for conv in self.convs:
            feats.append(self.act(conv(torch.cat(feats, 1))))
        return self.lff(torch.cat(feats, 1)) * self.beta + x


class RRDB(nn.Module):
    """Residual-in-residual dense block: 3 DenseBlocks, residual-scaled by beta."""

    def __init__(self, nf, gc=32, beta=0.2, n_dense=3):
        super().__init__()
        self.beta = beta
        self.blocks = nn.Sequential(*[DenseBlock(nf, gc) for _ in range(n_dense)])

    def forward(self, x):
        return self.blocks(x) * self.beta + x


class RRDBNet(nn.Module):
    """RRDB backbone (ESRGAN lineage), pre-upsampling residual variant for this
    harness (the standard sub-pixel upsampler would double-upsample the HR input)."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_basic=3,
                 n_dense=3, gc=32, scale=2, use_checkpoint=False, coord_channels=0, **_u):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.scale = int(scale)
        self.coord_channels = 0
        self.head = nn.Conv2d(self.in_channels, nf, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(nf, gc, 0.2, n_dense) for _ in range(n_basic)])
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.tail = nn.Conv2d(nf, self.out_channels, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        f = self.head(x)
        f = self.body_tail(self.body(f)) + f
        return self.tail(f)


def build_rrdb(size="S", **kw):
    kw.pop("patch_size", None)
    preset = {"XS": 32, "S": 64, "M": 96, "B": 128}
    if not kw.get("nf"):
        kw["nf"] = preset.get(str(size).upper(), 64)
    return RRDBNet(**kw)


# --------------------------------------------------------------- SRCNN / VDSR / RDN


class SRCNN(nn.Module):
    """Classic 3-layer CNN (Gupta et al., ICCV'15). It predicts the HR image itself
    (not a residual), so this arm is trained/evaluated with --residual 0. Its last
    conv is identity-initialised so step 0 equals the bicubic input exactly, which is
    what makes it comparable to the residual arms in the learnability gate."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, scale=2,
                 coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.head = nn.Conv2d(3, nf, 9, padding=4)
        self.net = nn.Sequential(
            self.head, nn.ReLU(True), nn.Conv2d(nf, nf, 5, padding=2), nn.ReLU(True),
            nn.Conv2d(nf, 3, 5, padding=2))
        with torch.no_grad():                      # net(x) == x at initialisation
            last = self.net[-1]
            last.weight.zero_(); last.bias.zero_()
            for c in range(3):
                last.weight[c, c, 2, 2] = 1.0
        self.align = 1

    def forward(self, x, t=None):
        return self.net(x)


class VDSR(nn.Module):
    """Very Deep SR (20 conv) with global residual — natively a residual predictor,
    matches the harness (pred = bicubic + net) exactly."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=64, n_res=18,
                 scale=2, coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.head = nn.Conv2d(3, nf, 3, padding=1)
        layers = []
        for _ in range(n_res):
            layers += [nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(True)]
        self.body = nn.Sequential(*layers)
        self.tail = nn.Conv2d(nf, 3, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        return self.tail(self.body(self.head(x)))


class RDB(nn.Module):
    def __init__(self, nf=16, gc=8, n_layers=5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(nf + i * gc, gc, 3, padding=1) for i in range(n_layers)])
        self.lff = nn.Conv2d(nf + n_layers * gc, nf, 1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        feats = [x]
        for c in self.convs:
            feats.append(self.act(c(torch.cat(feats, 1))))
        return self.lff(torch.cat(feats, 1)) + x


class RDN(nn.Module):
    """Residual Dense Network (pre-upsampling residual variant)."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=16, gc=8,
                 n_blocks=4, n_layers=5, scale=2, coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.head = nn.Conv2d(3, nf, 3, padding=1)
        self.body = nn.ModuleList([RDB(nf, gc, n_layers) for _ in range(n_blocks)])
        self.global_lff = nn.Conv2d(nf * (n_blocks + 1), nf, 1)
        self.tail = nn.Conv2d(nf, 3, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        x0 = self.head(x)
        feats = [x0]
        h = x0
        for blk in self.body:
            h = blk(h)
            feats.append(h)
        out = self.global_lff(torch.cat(feats, 1))
        return self.tail(out)


def build_srcnn(size="S", **kw):
    kw.pop("patch_size", None)
    return SRCNN(**kw)


def build_vdsr(size="S", **kw):
    kw.pop("patch_size", None)
    return VDSR(**kw)


def build_rdn(size="S", **kw):
    kw.pop("patch_size", None)
    kw.setdefault("nf", 16); kw.setdefault("gc", 8)
    return RDN(**kw)


# =============================================================================
#  Modern top-conference baselines (2022-2023): RLFN / HAT / SRFormer
#
#  Same faithfulness contract as EDSR/RCAN above: this harness feeds the
#  bicubic-upsampled HR image and adds our output back, so every net here is the
#  PRE-UPSAMPLING residual variant, trained from scratch at our sample budget.
#  They are honest "same-protocol representatives" of those architectures, not
#  bit-exact ports of the official repos (whose weights we cannot download here).
# =============================================================================


# ------------------------------------------------------------------------ RLFN

class RLFNLocalBlock(nn.Module):
    """Local feature block (RLFN, NTIRE'22 winner): 3 convs where each sees the
    block input plus the previous output, 1x1 local feature fusion, 0.1 scaling."""

    def __init__(self, nf, k=3):
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, k, padding=k // 2)
        self.c2 = nn.Conv2d(2 * nf, nf, k, padding=k // 2)
        self.c3 = nn.Conv2d(2 * nf, nf, k, padding=k // 2)
        self.lff = nn.Conv2d(4 * nf, nf, 1)

    def forward(self, x):
        x1 = F.relu(self.c1(x), True)
        x2 = F.relu(self.c2(torch.cat([x, x1], 1)), True)
        x3 = F.relu(self.c3(torch.cat([x, x2], 1)), True)
        return self.lff(torch.cat([x, x1, x2, x3], 1)) * 0.1 + x


class RLFNResDenseBlock(nn.Module):
    """The single residual dense block of RLFN (n layers, growth gc, 0.1 scaling)."""

    def __init__(self, nf, gc, n=8):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(nf + i * gc, gc, 3, padding=1) for i in range(n)])
        self.lff = nn.Conv2d(nf + n * gc, nf, 1)

    def forward(self, x):
        feats = [x]
        for c in self.convs:
            feats.append(F.relu(c(torch.cat(feats, 1)), True))
        return self.lff(torch.cat(feats, 1)) * 0.1 + x


class RLFN(nn.Module):
    """Residual Local Feature Network (Zhang et al., NTIRE'22 champion model).
    Config = official RLFN (nf48, 4+4 local blocks, 1 RDB x 8 layers, gc32, ~2.1M)."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, nf=48, n_res=4,
                 n_rdb=1, n_rdbnum=8, gc=32, scale=2, coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.head = nn.Conv2d(3, nf, 3, padding=1)
        self.lfb1 = nn.Sequential(*[RLFNLocalBlock(nf) for _ in range(n_res)])
        self.rdbs = nn.Sequential(*[RLFNResDenseBlock(nf, gc, n_rdbnum) for _ in range(n_rdb)])
        self.lfb2 = nn.Sequential(*[RLFNLocalBlock(nf) for _ in range(n_res)])
        self.tail = nn.Conv2d(nf, 3, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        f = self.head(x)
        f = self.lfb1(f)
        f = self.rdbs(f)
        f = self.lfb2(f)
        return self.tail(f)


def build_rlfN(size="S", **kw):
    kw.pop("patch_size", None)
    if kw.get("nf") is None:
        kw.pop("nf", None)
    if str(size).upper() == "XS":
        kw.setdefault("nf", 32)
    return RLFN(**kw)


# ---------------------------------------------------- shared window utilities

def win_partition(x, w):
    """(B, Hp, Wp, C) -> (B*nWin, w*w, C);  Hp/Wp must already be multiples of w."""
    b, hp, wp, c = x.shape
    x = x.view(b, hp // w, w, wp // w, w, c).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, w * w, c)


def win_reverse(x, w, hp, wp):
    """inverse of win_partition: (B*nWin, w*w, C) -> (B, hp, wp, C)."""
    b = x.shape[0] // ((hp // w) * (wp // w))
    x = x.view(b, hp // w, wp // w, w, w, -1).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, hp, wp, -1)


def rel_pos_index(w, device):
    """Swin-style relative position index for a w x w window -> (w², w²) into (2w-1)²."""
    coords = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing="ij"))  # (2,w,w)
    cf = torch.flatten(coords, 1)                                    # (2, w²)
    d = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()   # (N,N,2)
    idx = (d[..., 0] + w - 1) * (2 * w - 1) + (d[..., 1] + w - 1)
    return idx.to(device)


def pad_to_multiple(x, w):
    """x is (B, C, H, W); reflect-pad so H/W are multiples of w. Returns (x, pad_h, pad_w)."""
    b, c, h, wd = x.shape
    ph = (w - h % w) % w
    pw = (w - wd % w) % w
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw


# ------------------------------------------------------------------ SRFormer


def _permute_heads(x, nh, hd, sign):
    """x: (BN, N, nh, hd). Roll each head's token sequence by sign*(i+1)*hd.
    This is the parameter-free 'permuto' trick of SRFormer (Wang et al., ICCV'23):
    head i compares a query against keys displaced by a different offset, so the
    union of heads covers cross-position correspondences without extra params."""
    n = x.shape[1]
    outs = []
    for i in range(nh):
        sh = sign * ((i + 1) * hd) % n
        outs.append(torch.roll(x[:, :, i, :], shifts=sh, dims=1).unsqueeze(2))
    return torch.cat(outs, dim=2)


class PermutoAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.nh, self.hd, self.w = num_heads, dim // num_heads, window_size
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.rpe = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("rpe_idx", rel_pos_index(window_size, "cpu"), persistent=False)

    def forward(self, x):
        """x: (B*nWin, w*w, C) already windowed."""
        bn, n, c = x.shape
        qkv = self.qkv(x).view(bn, n, 3, self.nh, self.hd)
        q = _permute_heads(qkv[:, :, 0], self.nh, self.hd, +1).permute(0, 2, 1, 3) * self.scale
        k = _permute_heads(qkv[:, :, 1], self.nh, self.hd, -1).permute(0, 2, 3, 1)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        attn = q @ k
        bias = self.rpe[self.rpe_idx.reshape(-1)].view(n, n, self.nh).permute(2, 0, 1).unsqueeze(0)
        out = ((attn + bias).softmax(-1) @ v).transpose(1, 2)      # (bn, n, nh, hd)
        out = _permute_heads(out, self.nh, self.hd, -1)
        return self.proj(out.reshape(bn, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4, drop=0.0):
        super().__init__()
        hid = dim * ratio
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SRFormerBlock(nn.Module):
    """LN -> permuto-attention -> LN -> FFN, both branch outputs gated by a learned
    per-channel residual scale (tanh lambda, init 0), as in SRFormer/HAT."""

    def __init__(self, dim, num_heads, window_size, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PermutoAttention(dim, num_heads, window_size, proj_drop=drop, attn_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, mlp_ratio, drop)
        self.ls1 = nn.Parameter(torch.zeros(dim))
        self.ls2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x + torch.tanh(self.ls1) * self.attn(self.norm1(x))
        x = x + torch.tanh(self.ls2) * self.ffn(self.norm2(x))
        return x


class SRFormer(nn.Module):
    """SRFormer-S-style (dim 96, 5 groups x 6 blocks, window 16) in the harness's
    pre-upsampling residual form. No cross-window shift (so long-range context comes
    from the permute trick + the 3x3 conv between groups). HRA head-resizing is omitted."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, dim=96, depth=6,
                 groups=5, num_heads=6, window_size=16, mlp_ratio=4, scale=2,
                 coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.w = int(window_size)
        self.dim = int(dim)
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        layers, d = [], dim
        for _ in range(groups):
            layers += [SRFormerBlock(d, num_heads, window_size, mlp_ratio) for _ in range(depth)]
            layers += [nn.Conv2d(d, d, 3, padding=1)]      # between-group conv (NCHW)
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d)
        self.tail = nn.Conv2d(d, 3, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        b, _, h, wd = x.shape
        f, ph, pw = pad_to_multiple(self.head(x), self.w)
        hp, wp = f.shape[2], f.shape[3]
        win = win_partition(f.permute(0, 2, 3, 1), self.w)
        for m in self.layers:
            if isinstance(m, SRFormerBlock):
                win = m(win)
            else:                                            # conv between groups
                img = win_reverse(win, self.w, hp, wp).permute(0, 3, 1, 2)
                win = win_partition(m(img).permute(0, 2, 3, 1), self.w)
        img = win_reverse(self.norm(win), self.w, hp, wp).permute(0, 3, 1, 2)
        out = self.tail(img)
        return out[:, :, :h, :wd]


def build_srformer(size="S", **kw):
    kw.pop("patch_size", None)
    if str(size).upper() == "XS":
        kw.setdefault("dim", 64)
        kw.setdefault("num_heads", 4)
    return SRFormer(**kw)


# ---------------------------------------------------------------------- HAT


class ChannelAttentionBlock(nn.Module):
    """HAT's CAB: 5x5 depthwise conv pooling -> GAP -> FC -> ReLU -> FC -> sigmoid."""

    def __init__(self, dim, reduction=2):
        super().__init__()
        hid = max(dim // reduction, 8)
        self.dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.fc = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(True), nn.Linear(hid, dim), nn.Sigmoid())

    def forward(self, x):
        b, _, h, w, = x.shape
        y = F.adaptive_avg_pool2d(self.dw(x), 1).flatten(1)
        return x * self.fc(y).view(b, -1, 1, 1)


class WindowAttentionDW(nn.Module):
    """Window self-attention with relative position bias AND separate 3x3 depthwise
    convs on q/k/v (HAT's 'convolution in attention'). NOTE: HAT's ORPB uses an
    overlapped 3W x 3W context; we keep the non-overlapped window (documented in
    docs/baseline/README.md) because the official weights are not available offline."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, drop=0.0):
        super().__init__()
        self.nh, self.hd, self.w = num_heads, dim // num_heads, window_size
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dwq = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwk = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.rpe = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("rpe_idx", rel_pos_index(window_size, "cpu"), persistent=False)

    def _headify(self, t, bn, n):
        return t.view(bn, n, self.nh, self.hd).permute(0, 2, 1, 3)

    def forward(self, x):
        bn, n, c = x.shape
        w = self.w
        sp = self.qkv(x).view(bn, n, 3, c).permute(2, 0, 3, 1).reshape(3 * bn, c, w, w)
        q = self._headify(self.dwq(sp[:bn]).flatten(2).transpose(1, 2), bn, n)
        k = self._headify(self.dwk(sp[bn:2 * bn]).flatten(2).transpose(1, 2), bn, n)
        v = self._headify(self.dwv(sp[2 * bn:]).flatten(2).transpose(1, 2), bn, n)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        bias = self.rpe[self.rpe_idx.reshape(-1)].view(n, n, self.nh).permute(2, 0, 1).unsqueeze(0)
        out = (attn + bias).softmax(-1) @ v
        out = out.transpose(-1, -2).reshape(bn, n, c)
        return self.proj(out)


class CLAF(nn.Module):
    """Convolutional Lattice-Attention Fusion: per-branch 3x3 convs, gates =
    softmax over branches computed from a 3x3 conv of the concatenated features."""

    def __init__(self, dim, n_branch=2):
        super().__init__()
        self.nb = n_branch
        self.convs = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1) for _ in range(n_branch)])
        self.gate = nn.Conv2d(dim * n_branch, n_branch, 1)

    def forward(self, feats):
        cat = torch.cat(feats, 1)
        g = self.gate(cat).softmax(1)
        out = 0
        for i, conv in enumerate(self.convs):
            out = out + conv(feats[i]) * g[:, i:i + 1]
        return out


class HATBlock(nn.Module):
    """HAT hybrid-attention block: attn1(x) and attn2(CAB(x)) fused by a linear, plus a
    pure-conv path over [x, CAB(x)] (1x1 -> 3x3); the two are mixed by CLAF, then FFN.
    Both branches gated by learned tanh lambdas (init 0)."""

    def __init__(self, dim, num_heads, window_size, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = WindowAttentionDW(dim, num_heads, window_size, drop=drop)
        self.attn2 = WindowAttentionDW(dim, num_heads, window_size, drop=drop)
        self.cab = ChannelAttentionBlock(dim)
        self.fuse_attn = nn.Linear(dim * 2, dim)
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.claf = CLAF(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, mlp_ratio, drop)
        self.ls1 = nn.Parameter(torch.zeros(dim))
        self.ls2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        bn, n, c = x.shape
        w = int(round(n ** 0.5))
        assert w * w == n, "window token count must be a perfect square"
        f = self.norm1(x)
        img = f.view(bn, w, w, c).permute(0, 3, 1, 2)
        cab_img = self.cab(img)
        cab = cab_img.permute(0, 2, 3, 1).reshape(bn, n, c)
        y_attn = self.fuse_attn(torch.cat([self.attn1(f), self.attn2(cab)], -1))
        y_conv = self.conv3x3(self.fuse_conv(torch.cat([img, cab_img], 1)))
        y_attn = y_attn.view(bn, w, w, c).permute(0, 3, 1, 2)
        fused = self.claf([y_attn, y_conv]).permute(0, 2, 3, 1).reshape(bn, n, c)
        x = x + torch.tanh(self.ls1) * fused
        return x + torch.tanh(self.ls2) * self.ffn(self.norm2(x))


class HAT(nn.Module):
    """HAT-style hybrid-attention net (dim 96, 4 groups x 4 blocks, window 16) in the
    harness's pre-upsampling residual form. See WindowAttentionDW for the ORPB caveat."""

    def __init__(self, input_size=128, in_channels=3, out_channels=3, dim=96, depth=4,
                 groups=4, num_heads=6, window_size=16, mlp_ratio=4, scale=2,
                 coord_channels=0, use_checkpoint=False, **_u):
        super().__init__()
        self.w = int(window_size)
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        layers = []
        for _ in range(groups):
            layers += [HATBlock(dim, num_heads, window_size, mlp_ratio) for _ in range(depth)]
            layers += [nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Conv2d(dim, dim, 1)]
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)
        self.align = 1

    def forward(self, x, t=None):
        b, _, h, wd = x.shape
        f, ph, pw = pad_to_multiple(self.head(x), self.w)
        hp, wp = f.shape[2], f.shape[3]
        win = win_partition(f.permute(0, 2, 3, 1), self.w)
        for m in self.layers:
            if isinstance(m, HATBlock):
                win = m(win)
            else:
                img = win_reverse(win, self.w, hp, wp).permute(0, 3, 1, 2)
                win = win_partition(m(img).permute(0, 2, 3, 1), self.w)
        img = win_reverse(self.norm(win), self.w, hp, wp).permute(0, 3, 1, 2)
        return self.tail(img)[:, :, :h, :wd]


def build_hat(size="S", **kw):
    kw.pop("patch_size", None)
    if str(size).upper() == "XS":
        kw.setdefault("dim", 64)
        kw.setdefault("num_heads", 4)
    return HAT(**kw)


