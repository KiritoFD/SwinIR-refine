"""Train pixel-space DiT for RealSR x2 (condition = bicubic-up LR).

This is the "pixel" half of the 2x2 diffusion matrix:

* ``flow`` : rectified flow matching. Target is either HR or the residual
             ``(HR - bicubic_up) * 0.5`` (recommended: much smaller dynamic
             range, so the same number of ODE steps buys more accuracy).
* ``reg``  : deterministic residual regressor, ``HR = bicubic_up + f(bicubic_up)``
             with a zero-initialised head, L1 on pixels. This is the arm that is
             directly comparable to the SwinIR regression line (E11, Y 33.47).

HR crop = lr_patch * scale. Keep it at 128 (lr_patch 64) so it matches the
regression line's crop and the eval tile; 256 is feasible on 24G+ with
--grad-ckpt but is 4x the tokens.

Example:
  python -m diffusion.train_pixel --objective flow --lr-patch 64 --batch 8 \
    --steps 20000 --amp --grad-ckpt --out experiments/diffusion/pixel_flow
  python -m diffusion.train_pixel --objective reg --lr-patch 64 --batch 8 \
    --steps 20000 --amp --grad-ckpt --out experiments/diffusion/pixel_reg
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .data import RealSRCropDataset, make_split
from .data import default_root
from .dit import build_dit
from .mamba_sr import build_mambasr, mamba_ssm_available
from .unet import build_unet
from .unet import build_wavelet_dual
from .flow import flow_loss, sample_flow
from .metrics import official_pair_metrics
from .vae import psnr01, ssim01
from .wavelet import dwt_highfreq_loss
from .wavelet import dwt_hf_loss_shift
from .wavelet import dtcwt_hf_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--out", default=r"G:\RealSR\experiments\diffusion\pixel_dit")
    p.add_argument("--objective", default="flow", choices=["flow", "reg"])
    p.add_argument("--backbone", default="dit", choices=["dit", "unet", "mamba", "edsr", "rcan", "srresnet", "rrdb"],
                   help="dit = token transformer; unet = conv encoder/decoder with skips; "
                        "mamba = stride-1 VSS (2D selective scan), reg-only; edsr/rcan/srresnet/rrdb = "
                        "classic SR baselines (pre-upsampling residual variants of the canonical archs)")
    p.add_argument("--size", default="S", choices=["XS", "S", "M", "B"])
    p.add_argument("--patch", type=int, default=2)
    # U-Net only (ignored by the DiT path)
    p.add_argument("--base", type=int, default=0, help="unet: base channel count (0 = from --size)")
    p.add_argument("--mult", default="1,2,4,4", help="unet: channel multiplier per level")
    p.add_argument("--ffn", action="store_true",
                   help="add a spatial-gated FFN (continuous MoE gate) to every U-Net "
                        "residual block. Channel widths must then be 128-aligned, so use "
                        "--base 128 with it.")
    p.add_argument("--ffn-ratio", type=float, default=2.66)
    p.add_argument("--align", type=int, default=128,
                   help="channel alignment the gated FFN enforces. Must divide every "
                        "level's width, so base 64 needs align=64 while base 128 can "
                        "keep 128 (which is the one that runs at full GEMM efficiency).")
    p.add_argument("--coord", action="store_true",
                   help="concatenate absolute frame coordinates (2ch, [-1,1]^2) to the "
                        "input so the network can break conv translation invariance "
                        "(real lens degradation is non-stationary over the frame). "
                        "UNet reg only; stem weights on the coord channels are "
                        "zero-init, so step 0 is exactly the no-coord model.")
    p.add_argument("--freq-route", action="store_true",
                   help="per-ResBlock soft frequency routing: residual delta = "
                        "alpha*texture_conv + (1-alpha)*narrow smooth branch, "
                        "alpha = sigmoid(conv1x1(Sobel energy of the block input)). "
                        "The FFN's fix: capacity goes where high frequencies live.")
    p.add_argument("--dwt-unet", action="store_true",
                   help="UNet: replace the lossy stride-2 down / nearest up between "
                        "levels with a bijective orthogonal Haar DWT and its inverse. "
                        "Halves resolution while keeping every high-frequency bit, so it "
                        "aims to buy stride-1's phase fidelity at stride-2's cost and "
                        "receptive field. Requires --native-lr 0 (a stride-1 base); step 0 "
                        "is still exactly bicubic (zero-init output head).")
    p.add_argument("--dwt-dual", action="store_true",
                   help="UNet: wavelet DUAL-BRANCH net (Direction 1 done properly) -- one "
                        "Haar DWT, LL through a full U-Net at half res (lossless stride-2 "
                        "receptive field) + HL/LH/HH through a shallow branch, IWT recombined. "
                        "Requires --native-lr 0; step 0 == bicubic.")
    p.add_argument("--adv-deg", action="store_true",
                   help="with --pretrain-root: replace fixed BSRGAN degradation with "
                        "the differentiable adversary (diffusion.adversarial): "
                        "min_theta max_phi L(F(G_phi(hr)), hr). See that module.")
    p.add_argument("--adv-sigma-max", type=float, default=0.12,
                   help="noise ceiling of the adversary (in [0,1] RGB units). "
                        "0.12 ~= BSRGAN's max Gaussian sigma at x2.")
    p.add_argument("--adv-lr", type=float, default=1e-4, help="adversary (phi) lr")
    p.add_argument("--adv-inner", type=int, default=0,
                   help="0 = simultaneous GDA (one forward, phi ascends on the "
                        "flipped gradient). N>0 = N true inner ascent steps with "
                        "freshly regenerated LR before each theta step (N+1x cost).")
    p.add_argument("--num-res", type=int, default=2, help="unet: residual blocks per level; "
                   "mamba: VSS blocks per residual group")
    p.add_argument("--attn-levels", default="2,3", help="unet: levels that get self-attention")
    p.add_argument("--num-groups", type=int, default=4, help="mamba: residual groups")
    p.add_argument("--ssm-state", type=int, default=16, help="mamba: SSM state dim (d_state)")
    p.add_argument("--ssm-expand", type=float, default=2, help="mamba: channel expansion E")
    p.add_argument("--ssm-backend", default="auto", choices=["auto", "mamba_ssm", "torch"],
                   help="mamba: 'mamba_ssm' = fused CUDA kernel (if installed), "
                        "'torch' = pure-PyTorch chunked scan (exact, slower)")
    p.add_argument("--ssm-window", type=int, default=0,
                   help="mamba: windowed scan size. 0 = one global scan (L=H*W, very "
                        "slow at HR). 32 makes stride-1 affordable: the map is cut "
                        "into 32x32 windows that are scanned independently and "
                        "batched together, which is the same work but far more "
                        "parallel -- so the net can stay at stride 1.")
    p.add_argument("--ssm-shift", action="store_true",
                   help="mamba: alternate a padded half-window partition across "
                        "blocks so information crosses window boundaries.")
    p.add_argument("--ssm-scale", type=int, default=1, choices=[1, 2],
                   help="mamba: 1 = scan at HR (stride-1; ~30 s/step for 16 blocks at "
                        "HR128 -- measured), 2 = stride-2 stem, scan at LR (L 4x shorter, "
                        "MambaIR's own design), pixel-shuffle head back to HR")
    p.add_argument("--native-lr", type=int, default=-1,
                   help="unet: run the encoder/decoder at LR scale and pixel-shuffle back up. "
                        "-1 auto (on for reg, off for flow), 0 off, 1 on")
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--heads", type=int, default=0)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr-patch", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                   help="muon = Newton-Schulz orthogonalised momentum on >=2D weights "
                        "+ AdamW for the rest (reuses model/optim.Muon). A/B vs the "
                        "AdamW anchor; Muon wants its own lr (--muon-lr).")
    p.add_argument("--muon-lr", type=float, default=2e-3,
                   help="base lr when --optimizer muon (Muon converges at a higher lr "
                        "than AdamW's 3e-4); cosine-scheduled like the AdamW arm")
    p.add_argument("--muon-momentum", type=float, default=0.95,
                   help="Muon momentum (only with --optimizer muon)")
    p.add_argument("--muon-ns-steps", type=int, default=5,
                   help="Newton-Schulz orthogonalisation iterations (only with muon)")
    p.add_argument("--muon-aux-lr", type=float, default=0.0,
                   help="lr for the aux (bias/norm) AdamW inside the muon composite; 0 = same "
                        "as --muon-lr. Decoupling them lets the 1-D params train at a different "
                        "rate than the orthogonalised matrices.")
    p.add_argument("--overfit-test", action="store_true",
                   help="ORACLE ceiling probe: TRAIN ON THE Test split (all 100 pairs, "
                        "random crops) with pure L1 and no early stop.  Measures the "
                        "best Y any model can reach when it has memorised the answers -- "
                        "i.e. the dataset's representation + alignment + noise ceiling. "
                        "LEAKS the test set by design; a diagnostic, never a reported "
                        "result. Pair with --patience 999999 --min-steps 0 and NO --dwt-loss.")
    p.add_argument("--t-sampler", default="logit_normal")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--compile", action="store_true", help="torch.compile the training step")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--residual", type=int, default=1,
                   help="flow: model the residual (HR - bicubic_up) instead of HR")
    p.add_argument("--reg-loss", default="l1", choices=["l1", "l2", "smoothl1"])
    p.add_argument("--dwt-loss", action="store_true",
                   help="add an orthogonal Haar wavelet high-frequency subband L1 "
                        "(HL/LH/HH over --dwt-levels scales) to the reg objective. "
                        "Fights L1 mean-shrinkage / over-smoothing with a strictly "
                        "orthogonal, LOCAL frequency basis (unlike the failed FFT "
                        "band-gain heads). Training-only: zero inference cost, the same "
                        "eval_official protocol stays comparable. Backbone-agnostic (reg).")
    p.add_argument("--dwt-weight", type=float, default=1.0,
                   help="lambda on the wavelet high-frequency term (added to the main L1)")
    p.add_argument("--dwt-levels", type=int, default=2,
                   help="number of DWT decompositions (2 = level-1 + level-2 detail)")
    p.add_argument("--dwt-basis", default="haar", choices=["haar", "db2", "db4", "dtcwt"],
                   help="orthogonal analysis wavelet. haar = shift-sensitive 2-tap; "
                        "db2/db4 = smoother, better-localised Daubechies (still "
                        "orthonormal); dtcwt = dual-tree complex wavelet (2 offset trees "
                        "-> directional magnitude, ~shift-invariant).")
    p.add_argument("--dwt-w-hl", type=float, default=1.0, help="weight on the HL (vertical-edge) band")
    p.add_argument("--dwt-w-lh", type=float, default=1.0, help="weight on the LH (horizontal-edge) band")
    p.add_argument("--dwt-w-hh", type=float, default=1.0, help="weight on the HH (diagonal) band")
    p.add_argument("--dwt-w-ll", type=float, default=0.0,
                   help="optional weight re-adding the low-frequency (LL) subband term")
    p.add_argument("--dwt-shift", type=int, default=1,
                   help="average the wavelet-HF loss over this many dyadic shift offsets "
                        "{1,2,4}; >1 = shift-invariant (DTCWT-like), stops the net putting "
                        "detail in the wrong subband. 1 = the plain loss.")
    p.add_argument("--dwt-level-weights", default="",
                   help="comma list, one weight per DWT scale (e.g. '1.5,0.7' = emphasise "
                        "the coarse detail band, damp the fine one). '' = uniform")
    p.add_argument("--equiv", action="store_true",
                   help="D4 group-equivariance self-supervision: a second forward on a "
                        "randomly flipped/rot90'd input, penalising |T(model(x)) - "
                        "model(T(x))|. Makes the NET exactly equivariant rather than "
                        "only learning it in expectation from the (already present) "
                        "dataset augmentation. ~2x forward per step; inference unchanged.")
    p.add_argument("--equiv-weight", type=float, default=0.25,
                   help="alpha on the equivariance consistency term (proposal: 0.1-0.5)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--val-pairs", type=int, default=16)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-steps", type=int, default=4000)
    p.add_argument("--val-eval-steps", type=int, default=8)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-data", type=int, default=1,
                   help="preload and decode the training images once; without it the PNG "
                        "decode caps the loader at a few hundred samples/s")
    p.add_argument("--decoded-manifest", default="data/decoded/manifest.json",
                   help="manifest of pre-decoded uint8 blobs (scripts/server/precache_hr.py); "
                        "images found there are read via mmap instead of decoded. '' disables.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--init", default="",
                   help="load weights only (step/optimiser reset) — for fine-tuning a pretrained ckpt")
    p.add_argument("--pretrain-root", default="",
                   help="HR image dir for SwinIR/BSRGAN-style synthetic pretraining")
    p.add_argument("--pretrain-limit", type=int, default=0, help="cap pretraining images (0 = all)")
    p.add_argument("--pretrain-val-root", default="",
                   help="held-out images for the PRETRAIN val (must NOT overlap --pretrain-root). "
                        "Falls back to the RealSR split, which measures domain transfer rather "
                        "than pretraining progress.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not getattr(args, "data_root", ""):
        args.data_root = default_root()
    # the three new arms are UNet-reg features; keep invalid combos from
    # starting a 4-hour run only to die at step 0
    if args.coord and (args.backbone != "unet" or args.objective != "reg"):
        raise SystemExit("--coord requires --backbone unet --objective reg")
    if args.coord and args.pretrain_root:
        raise SystemExit("--coord + --pretrain-root is not wired (BSRGANDataset has no "
                         "frame coordinates yet); pretrain first, then --init finetune with --coord")
    if args.freq_route and args.backbone != "unet":
        raise SystemExit("--freq-route requires --backbone unet")
    if args.dwt_unet and args.backbone != "unet":
        raise SystemExit("--dwt-unet requires --backbone unet")
    if args.dwt_unet and int(args.native_lr) != 0:
        raise SystemExit("--dwt-unet requires --native-lr 0 (the wavelet replaces the "
                         "level downsampling; compounding it with a stride-2 stem is not "
                         "the intended arm)")
    if args.dwt_dual:
        if args.backbone != "unet" or args.objective != "reg":
            raise SystemExit("--dwt-dual requires --backbone unet --objective reg")
        if int(args.native_lr) != 0 or args.coord or args.freq_route or args.dwt_unet:
            raise SystemExit("--dwt-dual needs --native-lr 0 and no --coord/--freq-route/--dwt-unet")
    if args.backbone == "mamba" and args.objective != "reg":
        raise SystemExit("--backbone mamba is reg-only (the scan ignores t)")
    if args.adv_deg and not args.pretrain_root:
        raise SystemExit("--adv-deg requires --pretrain-root (it replaces BSRGAN there)")
    if args.dwt_loss and args.objective != "reg":
        raise SystemExit("--dwt-loss is a reg objective term (needs --objective reg)")
    if args.equiv and (args.backbone != "unet" or args.objective != "reg"):
        raise SystemExit("--equiv requires --backbone unet --objective reg")
    if args.equiv and args.adv_deg:
        raise SystemExit("--equiv + --adv-deg is not wired (adv replaces the LR batch)")
    return args


@torch.no_grad()
def run_val(net, val_ds, device, objective, steps, max_n, seed, residual):
    """True Y-PSNR on held-out pairs."""
    net.eval()
    ys, ss = [], []
    n = len(val_ds) if max_n <= 0 else min(len(val_ds), max_n)
    t0 = torch.zeros(1, device=device)
    coord_ch = int(getattr(net, "coord_channels", 0) or 0)
    for i in range(n):
        b = val_ds[i]
        lr = b["lr"].unsqueeze(0).to(device)
        hr = b["hr"].unsqueeze(0).to(device)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        inp = lr_up
        if coord_ch and b.get("coord") is not None:
            inp = torch.cat([lr_up, b["coord"].unsqueeze(0).float().to(device)], dim=1)
        if objective == "reg":
            rec = (lr_up + net(inp, t0)).clamp(0, 1)
        else:
            z = sample_flow(net, lr_up.shape, cond=inp, steps=steps, solver="heun", device=device, seed=seed)
            rec = (z * 2.0 + lr_up).clamp(0, 1) if residual else z.clamp(0, 1)
        sr_u8 = (rec[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        hr_u8 = (hr[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        m = official_pair_metrics(sr_u8, hr_u8)
        ys.append(m["psnr_y"])
        ss.append(m["ssim_y"])
    net.train()
    return float(np.mean(ys)), float(np.mean(ss))


@torch.no_grad()
def run_val_adv(net, adv, val_ds, device, max_n, seed):
    """Y-PSNR on held-out HR images degraded by the CURRENT adversary.

    Deterministic per item (fixed eps from the item index) so numbers are
    comparable across steps even though G_phi itself is moving.  This is the
    pretrain-domain val that drives ckpt_best in --adv-deg mode: robustness on
    the hardest degradation the adversary can currently express.
    """
    net.eval()
    ys = []
    n = len(val_ds) if max_n <= 0 else min(len(val_ds), max_n)
    t0 = torch.zeros(1, device=device)
    for i in range(n):
        b = val_ds[i]
        hr = b["hr"].unsqueeze(0).to(device)
        g = torch.Generator().manual_seed(seed * 100003 + int(b["idx"]))
        eps = torch.randn(1, 3, hr.shape[2] // adv.scale, hr.shape[3] // adv.scale,
                          generator=g).to(device)
        lr = adv(hr, eps=eps)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        rec = (lr_up + net(lr_up, t0)).clamp(0, 1)
        sr_u8 = (rec[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        hr_u8 = (hr[0] * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        ys.append(official_pair_metrics(sr_u8, hr_u8)["psnr_y"])
    net.train()
    return float(np.mean(ys))


def _worker_init(_worker_id: int):
    """Pin each DataLoader worker to a single compute thread.

    Left alone, every worker spawns its own OpenMP pool -- 24 threads each on
    this box, so 12 workers means ~288 spinning threads competing for 24 cores.
    BSRGAN's degradation is a handful of tiny 128x128 convolutions, and that
    thrash made it 10x slower than single-threaded: _blur 33.0 ms -> 3.3 ms,
    the full degrade 153 ms -> 10.6 ms per item.  That difference is the whole
    reason the loader could not feed a 768-sample batch.
    """
    torch.set_num_threads(1)


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def _geo(x: torch.Tensor, kind: int) -> torch.Tensor:
    """One D4 spatial element on a (B,C,H,W) image: 1 fliplr, 2 flipud, 3-5 rot90 k=1..3.

    Applied identically to the input and to the output so |T(f(x)) - f(T(x))| is a
    real equivariance test.  Patches are square (lr_patch x lr_patch) so rot90 keeps
    the shape; the stride-1 net's align divides H,W and rot90 preserves divisibility.
    """
    if kind == 1:
        return x.flip(-1)
    if kind == 2:
        return x.flip(-2)
    return torch.rot90(x, kind - 2, dims=(-2, -1))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dm = args.decoded_manifest or None
    _real = RealSRCropDataset(args.data_root, "Test" if args.overfit_test else "Train",
                              ("Canon", "Nikon"), args.scale, args.lr_patch, True,
                              cache=bool(args.cache_data), decoded_manifest=dm, coord=args.coord)
    if args.overfit_test:
        _real.augment = True  # random crops + flips so every step sees the whole frame
    adv = None
    if args.pretrain_root:
        # train on BSRGAN-degraded DIV2K/Flickr2K, but KEEP the RealSR val split
        # so we can watch zero-shot transfer while pretraining.
        if args.adv_deg:
            from .adversarial import HQPatchDataset

            ds = HQPatchDataset(args.pretrain_root, args.lr_patch * args.scale, True,
                                args.pretrain_limit, args.seed, decoded_manifest=dm)
            if args.pretrain_val_root:
                val_ds = HQPatchDataset(args.pretrain_val_root, args.lr_patch * args.scale, False,
                                        0, args.seed, decoded_manifest=dm, deterministic=True)
            else:
                _, val_ds = make_split(_real, args.val_pairs, seed=args.seed)
        else:
            from .bsrgan import BSRGANDataset

            ds = BSRGANDataset(args.pretrain_root, args.lr_patch, args.scale, True,
                               args.pretrain_limit, args.seed, decoded_manifest=dm)
            # Validate on HELD-OUT images from the pretrain domain, degraded the same
            # way.  Watching the RealSR val split during pretraining measures domain
            # transfer, not whether pretraining is working -- and since it drove
            # ckpt_best, the fine-tune was being initialised from whichever step
            # happened to transfer best, i.e. noise.
            if args.pretrain_val_root:
                val_ds = BSRGANDataset(args.pretrain_val_root, args.lr_patch, args.scale, False,
                                       0, args.seed, decoded_manifest=dm, deterministic=True)
            else:
                _, val_ds = make_split(_real, args.val_pairs, seed=args.seed)
        print(f"pretrain: {len(ds)} HR images from {args.pretrain_root}"
              f"{' (adversarial min-max)' if args.adv_deg else ' (BSRGAN)'}; "
              f"val {0 if val_ds is None else len(val_ds)} from "
              f"{args.pretrain_val_root or 'RealSR split (NOT recommended)'}", flush=True)
    else:
        if args.overfit_test:
            ds, val_ds = _real, _real  # val == train (leaked); patience set huge so no early stop
            print(f"  *** OVERFIT-TEST ORACLE: training on all {len(_real)} Test pairs, "
                  f"pure L1, no early stop.  Leakage by design -- ceiling probe only ***",
                  flush=True)
        else:
            ds, val_ds = make_split(_real, args.val_pairs, seed=args.seed)
    # RealSR Train has only 390 usable pairs.  With batch >= len(ds) an epoch
    # yields zero (drop_last) or one full batch, and the loop would rebuild the
    # worker pool on almost every step.  Draw several batches per epoch with
    # replacement instead, and keep the workers alive across epochs.
    sampler = None
    if len(ds) <= args.batch:
        sampler = RandomSampler(ds, replacement=True, num_samples=args.batch * 8)
        print(f"  sampler: {len(ds)} pairs < batch {args.batch} -> with replacement, "
              f"{args.batch * 8} samples/epoch ({8} steps)", flush=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init if args.num_workers > 0 else None,
    )
    hr_px = args.lr_patch * args.scale
    is_flow = args.objective == "flow"
    in_ch = 6 if is_flow else 3
    kw = {"input_size": hr_px, "patch_size": args.patch, "in_channels": in_ch, "use_checkpoint": args.grad_ckpt}
    if args.backbone in ("edsr", "rcan", "srresnet", "rrdb"):
        kw.pop("patch_size", None)
        kw["scale"] = int(args.scale)
        if args.base:
            kw["nf"] = int(args.base)
        from .baselines_sr import build_edsr, build_rcan, build_srresnet, build_rrdb
        _bmap = {"edsr": build_edsr, "rcan": build_rcan, "srresnet": build_srresnet, "rrdb": build_rrdb}
        model = _bmap[args.backbone](args.size, **kw).to(device)
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {args.backbone.upper()} nf={model.head.out_channels if hasattr(model.head,'out_channels') else '-'} "
              f"scale={args.scale} align={model.align} params={nparam:.2f}M "
              f"(classic pre-upsampling baseline, returns residual)", flush=True)
    elif args.backbone == "unet":
        kw.pop("patch_size")
        kw["mult"] = tuple(int(m) for m in args.mult.split(",") if m.strip())
        kw["num_res"] = args.num_res
        kw["attn_levels"] = tuple(int(m) for m in args.attn_levels.split(",") if m.strip())
        if args.base:
            kw["base"] = args.base
        ns = args.native_lr
        if ns < 0:
            ns = 1 if args.objective == "reg" else 0
        kw["in_stride"] = 2 if ns else 1
        kw["out_scale"] = 2 if ns else 1
        kw["ffn"] = bool(args.ffn)
        kw["ffn_ratio"] = float(args.ffn_ratio)
        kw["align"] = int(args.align)
        kw["coord_channels"] = 2 if args.coord else 0
        kw["freq_route"] = bool(args.freq_route)
        kw["wavelet"] = bool(args.dwt_unet)
        if args.dwt_dual:
            model = build_wavelet_dual(args.size, input_size=hr_px, in_channels=3,
                                       base=kw.get("base", 0), mult=kw["mult"],
                                       num_res=kw["num_res"], attn_levels=kw["attn_levels"],
                                       use_checkpoint=args.grad_ckpt).to(device)
            print(f"  unet DWT-DUAL align={model.align} chans={model.llnet.chans} "
                  f"(LL U-Net @ half-res + shallow HF branch)", flush=True)
        else:
            model = build_unet(args.size, **kw).to(device)
            print(f"  unet native_lr={bool(ns)} in_stride={kw['in_stride']} "
                  f"out_scale={kw['out_scale']} align={model.align} ffn={bool(args.ffn)} "
                  f"coord={bool(args.coord)} freq_route={bool(args.freq_route)} "
                  f"wavelet={bool(args.dwt_unet)} "
                  f"chans={model.chans}", flush=True)
    elif args.backbone == "mamba":
        if args.base:
            kw["dim"] = args.base
        kw["num_groups"] = args.num_groups
        kw["num_res"] = args.num_res
        kw["d_state"] = args.ssm_state
        kw["expand"] = args.ssm_expand
        kw["backend"] = args.ssm_backend
        kw["coord_channels"] = 2 if args.coord else 0
        kw["ssm_scale"] = args.ssm_scale
        kw["ssm_window"] = args.ssm_window
        kw["ssm_shift"] = bool(args.ssm_shift)
        model = build_mambasr(args.size, **kw).to(device)
        print(f"  mamba dim={model.dim} groups={args.num_groups} blocks/RG={args.num_res} "
              f"state={args.ssm_state} expand={args.ssm_expand} backend={model.backend} "
              f"scale={args.ssm_scale} window={args.ssm_window} shift={bool(args.ssm_shift)} "
              f"(mamba_ssm_available={mamba_ssm_available()}) coord={bool(args.coord)} "
              f"params={sum(q.numel() for q in model.parameters())/1e6:.2f}M", flush=True)
    else:
        if args.hidden:
            kw["hidden_size"] = args.hidden
        if args.depth:
            kw["depth"] = args.depth
        if args.heads:
            kw["num_heads"] = args.heads
        model = build_dit(args.size, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _name = {"unet": "PixelUNet", "dit": "PixelDiT", "mamba": "MambaSR",
             "edsr": "EDSR", "rcan": "RCAN", "srresnet": "SRResNet", "rrdb": "RRDB"}.get(args.backbone, args.backbone.upper())
    print(
        f"{_name}-{args.size} {n_params:.2f}M  "
        f"backbone={args.backbone} objective={args.objective}  HR={hr_px} "
        f"in_ch={in_ch} residual={bool(args.residual)} train={len(ds)} val={0 if val_ds is None else len(val_ds)}",
        flush=True,
    )
    if args.dwt_loss or args.equiv:
        print(f"  new-arms: dwt_loss={bool(args.dwt_loss)}(w={args.dwt_weight},"
              f"levels={args.dwt_levels}) equiv={bool(args.equiv)}(w={args.equiv_weight})",
              flush=True)

    model_raw = model
    is_muon = args.optimizer == "muon"
    base_lr = args.muon_lr if is_muon else args.lr
    if is_muon:
        from model.optim import build_optimizer
        opt = build_optimizer(model, name="muon", lr=base_lr, weight_decay=args.weight_decay,
                              momentum=args.muon_momentum, ns_steps=args.muon_ns_steps,
                              aux_lr=(args.muon_aux_lr if args.muon_aux_lr > 0 else None))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    # Muon returns a CompositeOptimizer (not a torch Optimizer) and bf16 needs no loss
    # scaling -> skip the GradScaler for muon; autocast still runs the forward in bf16.
    scaler = None if is_muon else torch.amp.GradScaler("cuda", enabled=args.amp)
    print(f"  optimizer={args.optimizer} lr0={base_lr:.2e} amp={bool(args.amp)} scaler={'off' if scaler is None else 'on'}", flush=True)

    adv = None
    opt_phi = None
    if args.adv_deg:
        from .adversarial import AdvDegradation

        adv = AdvDegradation(scale=args.scale, sigma_max=args.adv_sigma_max).to(device)
        opt_phi = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)
        print(f"  adversary: G_phi with {adv.n_kernels} bank kernels, "
              f"sigma_max={args.adv_sigma_max}, schedule="
              f"{'simultaneous GDA' if args.adv_inner == 0 else f'{args.adv_inner} inner steps'}",
              flush=True)

    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    if args.compile:
        try:
            model = torch.compile(model_raw, mode=args.compile_mode)
            print(f"torch.compile ON (mode={args.compile_mode})", flush=True)
        except Exception as exc:  # never let a compile failure kill a long run
            model = model_raw
            print(f"torch.compile FAILED ({exc}); falling back to eager", flush=True)

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters()):
                a.mul_(args.ema).add_(b.detach(), alpha=1 - args.ema)

    step = 0
    best = -1.0
    no_gain = 0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_raw.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step = int(ck.get("step", 0))
        best = float(ck.get("best_psnr_y", ck.get("best_psnr01", -1.0)))
        print(f"resumed from {args.resume} step={step} best={best:.3f}", flush=True)

    if args.init and Path(args.init).is_file():
        # weights only: fine-tune a pretrained ckpt from step 0 with a fresh schedule
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model_raw.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step, best, no_gain = 0, -1.0, 0
        print(f"init weights from {args.init} (training state reset)", flush=True)

    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    t_zeros = torch.zeros(1, device=device)

    def forward_loss(inp, lr_up, hr):
        if is_flow:
            x0 = (hr - lr_up).clamp(-1, 1) * 0.5 if args.residual else hr
            return flow_loss(model, x0, inp, t_mode=args.t_sampler)[0], None
        res = model(inp, t_zeros.expand(lr_up.shape[0]))
        pred = (lr_up + res).clamp(0, 1)
        if args.reg_loss == "l2":
            base = F.mse_loss(pred, hr)
        elif args.reg_loss == "smoothl1":
            base = F.smooth_l1_loss(pred, hr)
        else:
            base = F.l1_loss(pred, hr)
        if args.dwt_loss:
            lw = [float(x) for x in args.dwt_level_weights.split(",")] if args.dwt_level_weights else None
            if args.dwt_basis == "dtcwt":
                base = base + args.dwt_weight * dtcwt_hf_loss(pred, hr, levels=args.dwt_levels)
            else:
                common = dict(levels=args.dwt_levels, basis=args.dwt_basis,
                              band_w=(args.dwt_w_hl, args.dwt_w_lh, args.dwt_w_hh),
                              ll_w=args.dwt_w_ll, level_weights=lw)
                if args.dwt_shift > 1:
                    base = base + args.dwt_weight * dwt_hf_loss_shift(pred, hr, n_shift=args.dwt_shift, **common)
                else:
                    base = base + args.dwt_weight * dwt_highfreq_loss(pred, hr, **common)
        return base, pred

    t0 = time.time()
    it = iter(loader)
    model.train()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if adv is not None:
            hr = batch["hr"].to(device, non_blocking=True)
        else:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
        coord = batch["coord"].to(device, non_blocking=True) if args.coord else None
        cur_lr = lr_at(step, base_lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        # muon composite = [Muon matrices, AdamW 1-D]; when an aux lr is set, the aux
        # group must get its OWN cosine schedule (the uniform loop above would clobber it
        # with the muon lr).  groups[0]=muon, groups[1:]=aux.
        if is_muon and args.muon_aux_lr > 0:
            aux_lr = lr_at(step, args.muon_aux_lr, args.warmup, args.steps)
            for gi, g in enumerate(opt.param_groups):
                if gi > 0:
                    g["lr"] = aux_lr

        if adv is not None and args.adv_inner > 0:
            # true alternation: N real ascent steps for phi through the FROZEN
            # net, each with a freshly regenerated worst-case LR; then the outer
            # descent step below runs against the updated adversary
            for p in model_raw.parameters():
                p.requires_grad_(False)
            for _ in range(args.adv_inner):
                opt_phi.zero_grad(set_to_none=True)
                lr_i = adv(hr)
                lr_up_i = F.interpolate(lr_i, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    loss_i = forward_loss(lr_up_i, lr_up_i, hr)[0]
                scaler.scale(loss_i).backward()
                scaler.unscale_(opt_phi)
                for p in adv.parameters():
                    if p.grad is not None:
                        p.grad.neg_()
                torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
                scaler.step(opt_phi)
                scaler.update()
            for p in model_raw.parameters():
                p.requires_grad_(True)
            opt.zero_grad(set_to_none=True)
            lr = adv(hr)
        elif adv is not None:
            # simultaneous GDA: one forward/backward below serves both players
            # (phi ASCENDS by negating its grads before its step)
            lr = adv(hr)

        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        inp = torch.cat([lr_up, coord], dim=1) if coord is not None else lr_up
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            loss, pred1 = forward_loss(inp, lr_up, hr)
            if args.equiv:
                # one D4 element per step; consistency |T(pred1) - pred2|.  Grads
                # flow through both paths so the net is pushed toward exact
                # equivariance.  A single fixed transform per step still spans the
                # group across steps because the dataset already randomises D4.
                kind = int(torch.randint(1, 6, (1,)).item())
                res2 = model(_geo(inp, kind), t_zeros.expand(lr_up.shape[0]))
                pred2 = (lr_up + res2).clamp(0, 1)
                loss = loss + args.equiv_weight * F.l1_loss(_geo(pred1, kind), pred2)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if adv is not None:
                scaler.unscale_(opt_phi)
                for p in adv.parameters():
                    if p.grad is not None:
                        p.grad.neg_()
                torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
            scaler.step(opt)
            if adv is not None:
                scaler.step(opt_phi)
            scaler.update()
        else:
            # muon path (no GradScaler; bf16 forward, fp32 master weights, direct
            # step).  With an adversary present it must ALSO ascend on the flipped
            # grad -- muon+adv used to crash here because scaler was None.  opt_phi is
            # a plain Adam, so it needs no scaler.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if adv is not None:
                for p in adv.parameters():
                    if p.grad is not None:
                        p.grad.neg_()
                torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
                opt_phi.step()
                opt_phi.zero_grad(set_to_none=True)
            opt.step()
        opt.zero_grad(set_to_none=True)
        ema_update()

        if step % 50 == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
            print(
                f"step {step:06d}/{args.steps} | L {float(loss.detach()):.5f} | "
                f"lr {cur_lr:.2e} | {mem:.2f}GB | {time.time()-t0:.0f}s",
                flush=True,
            )
            with (out / "train_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": cur_lr}) + "\n")

        if args.eval_every > 0 and val_ds is not None and ((step + 1) % args.eval_every == 0):
            net = ema if ema is not None else model_raw
            if adv is not None and args.pretrain_val_root:
                psnr = run_val_adv(net, adv, val_ds, device, args.val_pairs, args.seed)
                ssim = 0.0
            else:
                psnr, ssim = run_val(
                    net, val_ds, device, args.objective, max(1, args.val_eval_steps),
                    args.val_pairs, args.seed, bool(args.residual),
                )
            print(f"  VAL {step+1}: Y {psnr:.3f}/{ssim:.4f} n={min(len(val_ds), args.val_pairs)}", flush=True)
            with (out / "val_log.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, "psnr_y": psnr, "ssim_y": ssim}) + "\n")
            if psnr > best:
                best = psnr
                no_gain = 0
                torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_best.pt")
            else:
                no_gain += 1
            if no_gain >= args.patience and step + 1 >= args.min_steps:
                print(f"  EARLY STOP at {step+1} (no val gain for {args.patience} evals)", flush=True)
                torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_last.pt")
                break

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save(save_payload(args, model, ema, step + 1, best, hr_px, in_ch), out / "ckpt_last.pt")
        step += 1

    print(f"done best_val_Y={best:.3f} → {out}", flush=True)


def _unwrap(m):
    """torch.compile wraps the module; keep checkpoint keys clean."""
    return getattr(m, "_orig_mod", m)


def save_payload(args, model, ema, step, best, hr_px, in_ch):
    net = ema if ema is not None else _unwrap(model)
    return {
        "step": step,
        "model": net.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "args": vars(args),
        "objective": args.objective,
        "mode": "pixel",
        "hr_px": hr_px,
        "in_channels": in_ch,
        "residual": bool(args.residual),
        "best_psnr_y": best,
        "best_psnr01": best,
    }


if __name__ == "__main__":
    main()
