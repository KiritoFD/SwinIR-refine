"""Official full-set RealSR eval for a trained latent/pixel DiT ckpt.

Handles all four arms of the diffusion matrix:

  mode=latent/pixel  x  objective=flow (ODE, --steps N) / reg (single forward)

Extras that matter for a comparable PSNR number:
  * deterministic initial noise (--seed) so run-to-run variance is 0
  * overlapping tiles with a linear cross-fade (no seam discontinuities)
  * --sweep-steps: PSNR vs NFE curve on a subset before the full run

Example:
  python -m diffusion.eval_official --ckpt .../ckpt_best.pt --vae flux1-vae \
    --steps 20 --tile 128 --pad 16 --sweep-steps 1,2,4,8,20,50 --sweep-pairs 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_index
from .data import default_root
from .dit import build_dit
from .unet import build_unet
from .unet import build_wavelet_dual
from .baselines_sr import build_edsr, build_rcan
from .flow import sample_flow
from .hf_flow import build_hf_head, hf_refine
from .iqa import IQAScorer
from .metrics import official_pair_metrics
from .vae import decode, encode, load_vae


def _amp(device, bf16: bool):
    """bf16 autocast for the DiT/flow sampling ONLY; the VAE stays fp32.

    Training (train_latent / train_pixel), the batch bench and the latent
    precompute all run the network in bf16, but this eval path was left in fp32
    -- twice the memory and roughly twice the wall clock of every other module
    in the repo.  The VAE is deliberately NOT autocast: a bf16 decode costs real
    PSNR, and the same split (bf16 model -> fp32 VAE decode) is what the
    reference implementation uses.

    Opt-in via --bf16 so that numbers produced before this change stay
    comparable.
    """
    if bf16 and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_rgb_u8(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


# ---- test-time augmentation (self-ensemble over the D4 group) ----------------
# 8 transforms = rot90 k=0..3 x optional horizontal flip.  A conv SR net is only
# approximately D4-equivariant (weight sharing + reflective padding break it at the
# borders), and bicubic upsampling IS equivariant, so averaging the 8 back-aligned
# predictions is a free variance-reduction / detail-consistency boost -- and for a
# zero-init (pure-bicubic) model it collapses exactly onto bicubic (smoke-checked).
_D4 = [(k, f) for k in range(4) for f in (0, 1)]


def _tta_fwd(a: np.ndarray, k: int, f: int) -> np.ndarray:
    b = np.rot90(a, k)
    if f:
        b = np.flip(b, axis=1)
    return np.ascontiguousarray(b)


def _tta_inv(y: np.ndarray, k: int, f: int) -> np.ndarray:
    if f:
        y = np.flip(y, axis=1)
    return np.ascontiguousarray(np.rot90(y, -k))


def sr_tta(base, lr_u8: np.ndarray, scale: int) -> np.ndarray:
    acc = None
    for k, f in _D4:
        inv = _tta_inv(base(_tta_fwd(lr_u8, k, f), scale), k, f)
        acc = inv.astype(np.float32) if acc is None else acc + inv.astype(np.float32)
    return np.clip(np.rint(acc / len(_D4)), 0, 255).astype(np.uint8)


def _grid(n: int, core: int) -> list[int]:
    if n <= core:
        return [0]
    ys = list(range(0, n - core + 1, core))
    if ys[-1] + core < n:
        ys.append(n - core)
    return ys


def _window(n: int, ramp: int, has_left: bool, has_right: bool, device) -> torch.Tensor:
    """Linear ramp of width `ramp` at the tile ends; 1 elsewhere.

    Two neighbouring tiles' ramps overlap by exactly `ramp` rows and sum to ~1,
    which removes the hard seam you get from plain averaging.
    """
    w = torch.ones(n, device=device, dtype=torch.float32)
    if ramp > 0 and (has_left or has_right):
        d = torch.arange(n, device=device, dtype=torch.float32)
        if has_left:
            w = torch.minimum(w, d / float(ramp))
        if has_right:
            w = torch.minimum(w, (n - 1 - d) / float(ramp))
    return w.clamp_min(0.0)


@torch.no_grad()
def _accumulate(out, acc, rec, H, W, pad, y0, x0):
    """Accumulate the whole padded tile `rec` (which starts at HR y0,x0) into out/acc.

    Neighbouring tiles overlap by 2*pad and their ramps sum to ~1, so the seam
    disappears instead of showing up as a hard cut in the PSNR map.
    """
    rh, rw = rec.shape[-2], rec.shape[-1]
    wy = _window(rh, 2 * pad, has_left=(y0 > 0), has_right=(y0 + rh < H), device=rec.device)
    wx = _window(rw, 2 * pad, has_left=(x0 > 0), has_right=(x0 + rw < W), device=rec.device)
    w = wy[:, None] * wx[None, :]
    out[:, :, y0 : y0 + rh, x0 : x0 + rw] += rec * w
    acc[:, :, y0 : y0 + rh, x0 : x0 + rw] += w


def _tile_windows(H, W, core, pad, align):
    """(y0, x0, y1, x1) extents; cores sit on a `core` grid with `pad` context around."""
    wins = []
    for y in _grid(H, core):
        for x in _grid(W, core):
            y1 = min(H, y + core + pad)
            x1 = min(W, x + core + pad)
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y0 -= y0 % align
            x0 -= x0 % align
            y1 = min(H, y1 + (align - y1 % align) % align)
            x1 = min(W, x1 + (align - x1 % align) % align)
            wins.append((y0, x0, y1, x1))
    return wins


def _chunked(idx, n):
    for i in range(0, len(idx), n):
        yield idx[i : i + n]


@torch.no_grad()
def sr_latent_tiled(model, vae, vinfo, lr_u8, scale, steps, tile, pad, device, objective, seed,
                    tile_batch=8, residual=False, bf16=False):
    """Bicubic-up LR -> tile encode -> flow ODE (or single reg forward) -> decode -> stitch."""
    h, w = lr_u8.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    ds = vinfo.downscale
    H = hr_h + (ds - hr_h % ds) % ds
    W = hr_w + (ds - hr_w % ds) % ds
    out = torch.zeros(1, 3, H, W, device=device)
    acc = torch.zeros(1, 1, H, W, device=device)

    lr_t = to_tensor(lr_u8).unsqueeze(0).to(device)
    # upscale to the EXACT HR size, then edge-pad to the VAE grid.
    # (Interpolating straight to the padded size would stretch the image and
    #  cost several dB of misalignment — measured: 0.47 dB vs 4.9 dB.)
    lr_up = F.interpolate(lr_t, size=(hr_h, hr_w), mode="bicubic", align_corners=False).clamp(0, 1)
    if H != hr_h or W != hr_w:
        lr_up = F.pad(lr_up, (0, W - hr_w, 0, H - hr_h), mode="replicate")

    core = tile * scale                      # HR core == the crop the model was trained on
    core = max(core - core % ds, ds * 8)
    wins = _tile_windows(H, W, core, pad, ds)
    t0 = torch.zeros(1, device=device)

    # group equal-sized windows so they can be pushed through as one batch
    by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for win in wins:
        by_shape.setdefault((win[2] - win[0], win[3] - win[1]), []).append(win)

    for shape, group in by_shape.items():
        for chunk in _chunked(group, max(1, tile_batch)):
            cond = torch.cat([lr_up[:, :, y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk], dim=0)
            zc = encode(vae, cond, vinfo, sample=False)
            with _amp(device, bf16):
                if objective == "reg":
                    z = zc + model(zc, t0.expand(zc.shape[0]))
                else:
                    d = sample_flow(
                        model,
                        (zc.shape[0], vinfo.latent_channels, zc.shape[-2], zc.shape[-1]),
                        cond=zc,
                        steps=steps,
                        solver="heun",
                        device=device,
                        seed=seed,
                    )
                    z = zc + d if residual else d
            rec = decode(vae, z.float(), vinfo)
            for i, (y0, x0, y1, x1) in enumerate(chunk):
                _accumulate(out, acc, rec[i : i + 1], H, W, pad, y0, x0)
            del cond, zc, z, rec
    out = out / acc.clamp(min=1e-6)
    return to_u8(out[0, :, :hr_h, :hr_w])


@torch.no_grad()
def sr_pixel_tiled(model, lr_u8, scale, steps, tile, pad, residual, device, objective, seed,
                   tile_batch=8, bf16=False, hf_head=None, hf_steps=4, hf_scale=1.0):
    h, w = lr_u8.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    # a U-Net with in_stride/out_scale needs tiles aligned to 2^(levels+1);
    # the DiT only needs patch alignment (2).
    align = max(2, int(getattr(model, "align", 2) or 2))
    # RealSR HR sizes are not multiples of `align` (e.g. 1000 % 32 == 8), and
    # _tile_windows clamps the last window to H, which would leave the U-Net
    # dividing an unaligned size and its skip concat failing.  Pad the image up
    # to a multiple of align, work there, crop the result back -- same trick the
    # latent path already uses for the VAE grid.
    H = hr_h + (align - hr_h % align) % align
    W = hr_w + (align - hr_w % align) % align
    out = torch.zeros(1, 3, H, W, device=device)
    acc = torch.zeros(1, 1, H, W, device=device)
    lr_t = to_tensor(lr_u8).unsqueeze(0).to(device)
    lr_up = F.interpolate(lr_t, size=(hr_h, hr_w), mode="bicubic", align_corners=False).clamp(0, 1)
    if H != hr_h or W != hr_w:
        lr_up = F.pad(lr_up, (0, W - hr_w, 0, H - hr_h), mode="replicate")
    coord_map = None
    if int(getattr(model, "coord_channels", 0) or 0):
        # absolute frame coordinates over the PADDED canvas; padded rows/cols
        # clamp to the last valid index, matching the image's replicate padding.
        # Each tile window slices its own sub-range, so a tile sees the SAME
        # coordinates it would have seen inside the full frame during training.
        ys = torch.arange(H, device=device).float().clamp_max(max(hr_h - 1, 1))
        xs = torch.arange(W, device=device).float().clamp_max(max(hr_w - 1, 1))
        vy = ys / max(hr_h - 1, 1) * 2.0 - 1.0
        ux = xs / max(hr_w - 1, 1) * 2.0 - 1.0
        coord_map = torch.stack(
            [vy[:, None].expand(H, W), ux[None, :].expand(H, W)], dim=0
        ).unsqueeze(0)  # (1,2,H,W)
    core = max(tile * scale - (tile * scale) % align, align)
    wins = _tile_windows(H, W, core, pad, align)
    t0 = torch.zeros(1, device=device)

    by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for win in wins:
        by_shape.setdefault((win[2] - win[0], win[3] - win[1]), []).append(win)

    for shape, group in by_shape.items():
        for chunk in _chunked(group, max(1, tile_batch)):
            rgb = torch.cat([lr_up[:, :, y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk], dim=0)
            cond = rgb
            if coord_map is not None:
                cw = torch.cat([coord_map[:, :, y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk], dim=0)
                cond = torch.cat([rgb, cw], dim=1)
            with _amp(device, bf16):
                if objective == "reg":
                    rec = rgb + model(cond, t0.expand(cond.shape[0]))
                else:
                    z = sample_flow(
                        model,
                        (cond.shape[0], 3, cond.shape[-2], cond.shape[-1]),
                        cond=cond,
                        steps=steps,
                        solver="heun",
                        device=device,
                        seed=seed,
                    )
                    rec = (z * 2.0 + rgb) if residual else z
            # the fp32 `rgb` promotes the sum back to fp32 automatically
            rec = rec.clamp(0, 1).float()
            for i, (y0, x0, y1, x1) in enumerate(chunk):
                _accumulate(out, acc, rec[i : i + 1], H, W, pad, y0, x0)
            del cond, rgb, rec
    out = out / acc.clamp(min=1e-6)
    if hf_head is not None:
        # out is the full padded float prediction (1,3,H,W) with H,W multiples of
        # align -> even, so hf_refine's Haar DWT is well defined; LL untouched.
        out = hf_refine(hf_head, out, steps=hf_steps, device=device, seed=seed, scale=hf_scale)
    return to_u8(out[0, :, :hr_h, :hr_w])


def run_pairs(pairs, sr_fn, shave=0, oom_state=None, iqa=None):
    """sr_fn(lr_u8, scale) -> uint8 SR. On OOM, halves tile_batch and retries once."""
    rows = []
    for i, (lr_path, hr_path, sc) in enumerate(pairs):
        lr_u8 = load_rgb_u8(lr_path)
        hr_u8 = load_rgb_u8(hr_path)
        try:
            sr = sr_fn(lr_u8, sc)
        except RuntimeError as e:  # noqa: BLE001
            if "out of memory" in str(e).lower() and oom_state is not None and oom_state["tb"] > 1:
                oom_state["tb"] = max(1, oom_state["tb"] // 2)
                torch.cuda.empty_cache()
                print(f"  OOM -> tile_batch={oom_state['tb']}", flush=True)
                try:
                    sr = sr_fn(lr_u8, sc)
                except Exception as e2:  # noqa: BLE001
                    print(f"  FAIL {Path(lr_path).name}: {e2}", flush=True)
                    continue
            else:
                print(f"  FAIL {Path(lr_path).name}: {e}", flush=True)
                continue
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {Path(lr_path).name}: {e}", flush=True)
            continue
        hh = min(sr.shape[0], hr_u8.shape[0])
        ww = min(sr.shape[1], hr_u8.shape[1])
        a, b = sr[:hh, :ww], hr_u8[:hh, :ww]
        if shave > 0:
            a = a[shave:-shave, shave:-shave]
            b = b[shave:-shave, shave:-shave]
        m = official_pair_metrics(a, b)
        if iqa is not None:
            # NR-IQA on the SR image only -- this is the half PSNR cannot see
            m.update(iqa.score(a))
        m["name"] = Path(lr_path).name
        rows.append(m)
        extra = ""
        if iqa is not None:
            extra = "  " + " ".join(
                f"{k} {m[k]:.3f}" for k in iqa.available() if k in m
            )
        print(
            f"  [{i+1}/{len(pairs)}] {m['name']} Y {m['psnr_y']:.2f}/{m['ssim_y']:.4f} "
            f"RGB {m['psnr_rgb']:.2f}{extra}",
            flush=True,
        )
    return rows


def mean(rows, k):
    # the guard has to be `k in r`, not `r.get(k) == r.get(k)`: for a missing key
    # that comparison is None == None -> True, and the r[k] then raises KeyError.
    # It bites as soon as any optional column (e.g. the IQA scores) is absent.
    vals = [r[k] for r in rows if k in r and r[k] == r[k]]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default="", help="auto-detected if empty")
    p.add_argument("--vae", default="", help="required for latent mode")
    p.add_argument("--mode", default="auto", choices=["auto", "latent", "pixel"])
    p.add_argument("--objective", default="auto", choices=["auto", "flow", "reg"])
    p.add_argument("--cameras", default="Canon,Nikon")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--steps", type=int, default=20, help="flow: ODE steps (NFE = 2*steps for heun)")
    p.add_argument("--tile", type=int, default=128, help="LR core size; HR core = tile*scale")
    p.add_argument("--pad", type=int, default=16, help="HR-pixel context margin around each tile")
    p.add_argument("--shave", type=int, default=0, help="border shave before metrics (official uses 0)")
    p.add_argument("--seed", type=int, default=1234, help="flow: fixed initial noise")
    p.add_argument("--tile-batch", type=int, default=0,
                   help="tiles pushed through the model at once (0 -> 8). Bigger = much faster eval")
    p.add_argument("--sweep-steps", default="", help="e.g. 1,2,4,8,20,50 — PSNR vs NFE on a subset")
    p.add_argument("--sweep-pairs", type=int, default=8)
    p.add_argument("--out", default="")
    p.add_argument("--residual", action="store_true", help="pixel flow: HR = residual + bicubic-up")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iqa", action="store_true",
                   help="also compute MUSIQ / MANIQA (no-reference perceptual). "
                        "Slower, and needs the pyiqa checkpoints pre-placed.")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for the DiT/flow only (the VAE stays fp32). "
                        "Roughly 2x faster and 2x less memory.  Off by default so "
                        "that numbers produced before this flag stay comparable.")
    p.add_argument("--tta", action="store_true",
                   help="self-ensemble over the 8-element D4 group (rot90 x flip): run SR on "
                        "each transform, back-align and average in float. ~8x eval cost, no "
                        "retraining; usually a free PSNR + perceptual gain. Off by default so "
                        "non-TTA numbers stay comparable.")
    p.add_argument("--hf-head", default="",
                   help="residual HF rectified-flow head ckpt (diffusion.train_hf_flow). "
                        "When set, the regression SR is refined in the Haar HF subbands "
                        "before quantisation (LL untouched -> fidelity-safe, perception-targeted).")
    p.add_argument("--hf-steps", type=int, default=4, help="HF flow sampling NFE")
    p.add_argument("--hf-scale", type=float, default=1.0,
                   help="how much generated HF detail to add (dial perception vs fidelity)")
    args = p.parse_args()
    if not getattr(args, "data_root", ""):
        args.data_root = default_root()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ck.get("args", {}) or {}
    mode = args.mode
    if mode == "auto":
        mode = ck.get("mode") or ("latent" if ("latent_size" in ck or targs.get("vae")) else "pixel")
    objective = args.objective
    if objective == "auto":
        objective = ck.get("objective") or targs.get("objective") or "flow"

    in_ch = int(ck.get("in_channels") or targs.get("in_channels") or (32 if mode == "latent" else 6))
    size = targs.get("size", "S")
    patch = int(targs.get("patch", 2))
    residual = bool(args.residual or targs.get("residual", False))

    vae = vinfo = None
    if mode == "latent":
        vae_name = args.vae or ck.get("vae") or targs.get("vae") or "flux1-vae"
        vae, vinfo = load_vae(vae_name, device=device)
        input_size = int(ck.get("latent_size") or (targs.get("lr_patch", 128) * targs.get("scale", 2) // vinfo.downscale))
        default_tile = int(targs.get("lr_patch", 128))
    else:
        input_size = int(ck.get("hr_px") or targs.get("lr_patch", 64) * targs.get("scale", 2))
        default_tile = input_size // int(targs.get("scale", 2))

    def _ints(s, dflt):
        parts = [x for x in str(s if s is not None else dflt).split(",") if x.strip()]
        return tuple(int(x) for x in parts) or tuple(int(x) for x in dflt.split(","))

    backbone = str(targs.get("backbone", "dit"))
    use_coord = bool(targs.get("coord", False)) and mode == "pixel"
    if use_coord and objective == "flow":
        raise SystemExit("this ckpt was trained with --coord, which is reg-only")
    if backbone in ("edsr", "rcan"):
        _f = build_edsr if backbone == "edsr" else build_rcan
        model = _f(size, input_size=input_size, in_channels=in_ch,
                   nf=int(targs.get("base", 0) or 0) or None,
                   scale=int(targs.get("scale", 2))).to(device)
    elif backbone == "unet":
        # native_lr was -1 (auto) unless explicitly set; mirror train_pixel's rule
        ns = int(targs.get("native_lr", -1))
        if ns < 0:
            ns = 1 if objective == "reg" else 0
        if bool(targs.get("dwt_dual", False)):
            model = build_wavelet_dual(
                size, input_size=input_size, in_channels=3,
                base=int(targs.get("base", 0) or 0),
                mult=_ints(targs.get("mult"), "1,2,4,4"),
                num_res=int(targs.get("num_res", 2)),
                attn_levels=_ints(targs.get("attn_levels"), "2,3"),
            ).to(device)
        else:
            model = build_unet(
                size, input_size=input_size, in_channels=in_ch,
                base=int(targs.get("base", 0) or 0),
                mult=_ints(targs.get("mult"), "1,2,4,4"),
                num_res=int(targs.get("num_res", 2)),
                attn_levels=_ints(targs.get("attn_levels"), "2,3"),
                in_stride=2 if ns else 1,
                out_scale=2 if ns else 1,
                ffn=bool(targs.get("ffn", False)),
                ffn_ratio=float(targs.get("ffn_ratio", 2.66)),
                align=int(targs.get("align", 128)),
                coord_channels=2 if use_coord else 0,
                freq_route=bool(targs.get("freq_route", False)),
                wavelet=bool(targs.get("dwt_unet", False)),
            ).to(device)
    elif backbone == "mamba":
        from .mamba_sr import build_mambasr

        model = build_mambasr(
            size, input_size=input_size, in_channels=in_ch,
            dim=int(targs.get("base", 0) or 0) or None,
            num_groups=int(targs.get("num_groups", 4)),
            num_res=int(targs.get("num_res", 4)),
            d_state=int(targs.get("ssm_state", 16)),
            expand=float(targs.get("ssm_expand", 2)),
            backend=str(targs.get("ssm_backend", "auto")),
            coord_channels=2 if use_coord else 0,
            ssm_scale=int(targs.get("ssm_scale", 1)),
            ssm_window=int(targs.get("ssm_window", 0)),
            ssm_shift=bool(targs.get("ssm_shift", False)),
        ).to(device)
    else:
        model = build_dit(size, input_size=input_size, patch_size=patch, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if args.tile <= 0:
        args.tile = default_tile
    print(
        f"loaded {ckpt_path} mode={mode} objective={objective} backbone={backbone} in_ch={in_ch} size={size} "
        f"input={input_size} tile={args.tile} pad={args.pad} seed={args.seed}",
        flush=True,
    )

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pairs = build_index(args.data_root, cameras, "Test", args.scale)
    if args.max_pairs and args.max_pairs > 0:
        step = max(1, len(pairs) // args.max_pairs)
        pairs = pairs[::step][: args.max_pairs]
    print(f"pairs: {len(pairs)}", flush=True)

    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval_official"
    out_dir.mkdir(parents=True, exist_ok=True)

    oom_state = {"tb": args.tile_batch if args.tile_batch > 0 else 8}

    hf_head = None
    if args.hf_head:
        hc = torch.load(args.hf_head, map_location="cpu", weights_only=False)
        ha = hc.get("args", {}) or {}
        hf_head = build_hf_head("S", base=int(ha.get("head_base", 64)), ch=3,
                                mult=tuple(int(x) for x in str(ha.get("head_mult", "1,2,4")).split(",")),
                                num_res=int(ha.get("head_res", 2)), attn_levels=()).to(device)
        hf_head.load_state_dict(hc["model"])
        hf_head.eval()
        print(f"  HF flow head: {args.hf_head} (steps={args.hf_steps} scale={args.hf_scale})", flush=True)

    def make_sr_fn(steps):
        if mode == "latent":
            base = lambda lr_u8, sc: sr_latent_tiled(
                model, vae, vinfo, lr_u8, sc, steps, args.tile, args.pad, device, objective, args.seed,
                tile_batch=oom_state["tb"], residual=residual, bf16=args.bf16,
            )
        else:
            base = lambda lr_u8, sc: sr_pixel_tiled(
                model, lr_u8, sc, steps, args.tile, args.pad, residual, device, objective, args.seed,
                tile_batch=oom_state["tb"], bf16=args.bf16,
                hf_head=hf_head, hf_steps=args.hf_steps, hf_scale=args.hf_scale,
            )
        if args.tta:
            return lambda lr_u8, sc: sr_tta(base, lr_u8, sc)
        return base

    sweep = []
    if args.sweep_steps and objective != "reg":
        sub = pairs[:: max(1, len(pairs) // max(args.sweep_pairs, 1))][: args.sweep_pairs]
        for s in [int(x) for x in args.sweep_steps.split(",") if x.strip()]:
            rows = run_pairs(sub, make_sr_fn(s), args.shave, oom_state)
            rec = {"steps": s, "nfe": 2 * s, "n": len(rows), "psnr_y": mean(rows, "psnr_y"), "ssim_y": mean(rows, "ssim_y")}
            sweep.append(rec)
            print(f"SWEEP steps={s} nfe={2*s} n={rec['n']} Y {rec['psnr_y']:.3f}/{rec['ssim_y']:.4f}", flush=True)

    iqa = IQAScorer(device) if args.iqa else None
    if iqa is not None:
        iqa.available()
    rows = run_pairs(pairs, make_sr_fn(args.steps), args.shave, oom_state, iqa=iqa)

    summary = {
        "protocol": "RealSR official Test.m (limited-range Y, uint8, shave=0; same as model/eval.py)",
        "ckpt": str(ckpt_path),
        "mode": mode,
        "objective": objective,
        "n": len(rows),
        "psnr_y": mean(rows, "psnr_y"),
        "ssim_y": mean(rows, "ssim_y"),
        "psnr_rgb": mean(rows, "psnr_rgb"),
        "steps": args.steps,
        "tile": args.tile,
        "pad": args.pad,
        "tta": bool(args.tta),
        "seed": args.seed,
        "sweep": sweep,
        # whatever the IQA scorer actually loaded (names are pyiqa metric ids)
        "iqa_scores": {n: mean(rows, n) for n in (iqa.available() if iqa else [])},
        "per_image": rows,
    }
    path = out_dir / "eval.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n== OFFICIAL n={summary['n']} mode={mode}/{objective} steps={args.steps} | "
        f"Y {summary['psnr_y']:.4f}/{summary['ssim_y']:.4f} | RGB {summary['psnr_rgb']:.2f}"
        + (" | " + " ".join(f"{k} {v:.3f}" for k, v in summary["iqa_scores"].items())
           if args.iqa and summary["iqa_scores"] else "") + " ==",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
