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
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_index
from .data import default_root
from .dit import build_dit
from .flow import sample_flow
from .metrics import official_pair_metrics
from .vae import decode, encode, load_vae


def load_rgb_u8(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


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
                    tile_batch=8):
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
            if objective == "reg":
                z = zc + model(zc, t0.expand(zc.shape[0]))
            else:
                z = sample_flow(
                    model,
                    (zc.shape[0], vinfo.latent_channels, zc.shape[-2], zc.shape[-1]),
                    cond=zc,
                    steps=steps,
                    solver="heun",
                    device=device,
                    seed=seed,
                )
            rec = decode(vae, z, vinfo)
            for i, (y0, x0, y1, x1) in enumerate(chunk):
                _accumulate(out, acc, rec[i : i + 1], H, W, pad, y0, x0)
            del cond, zc, z, rec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = out / acc.clamp(min=1e-6)
    return to_u8(out[0, :, :hr_h, :hr_w])


@torch.no_grad()
def sr_pixel_tiled(model, lr_u8, scale, steps, tile, pad, residual, device, objective, seed,
                   tile_batch=8):
    h, w = lr_u8.shape[:2]
    hr_h, hr_w = h * scale, w * scale
    out = torch.zeros(1, 3, hr_h, hr_w, device=device)
    acc = torch.zeros(1, 1, hr_h, hr_w, device=device)
    lr_t = to_tensor(lr_u8).unsqueeze(0).to(device)
    lr_up = F.interpolate(lr_t, size=(hr_h, hr_w), mode="bicubic", align_corners=False).clamp(0, 1)
    core = max(tile * scale - (tile * scale) % 2, 2)
    wins = _tile_windows(hr_h, hr_w, core, pad, 2)
    t0 = torch.zeros(1, device=device)

    by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for win in wins:
        by_shape.setdefault((win[2] - win[0], win[3] - win[1]), []).append(win)

    for shape, group in by_shape.items():
        for chunk in _chunked(group, max(1, tile_batch)):
            cond = torch.cat([lr_up[:, :, y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk], dim=0)
            if objective == "reg":
                rec = (cond + model(cond, t0.expand(cond.shape[0]))).clamp(0, 1)
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
                rec = (z * 2.0 + cond).clamp(0, 1) if residual else z.clamp(0, 1)
            for i, (y0, x0, y1, x1) in enumerate(chunk):
                _accumulate(out, acc, rec[i : i + 1], hr_h, hr_w, pad, y0, x0)
            del cond, rec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = out / acc.clamp(min=1e-6)
    return to_u8(out[0])


def run_pairs(pairs, sr_fn, shave=0, oom_state=None):
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
        m["name"] = Path(lr_path).name
        rows.append(m)
        print(
            f"  [{i+1}/{len(pairs)}] {m['name']} Y {m['psnr_y']:.2f}/{m['ssim_y']:.4f} RGB {m['psnr_rgb']:.2f}",
            flush=True,
        )
    return rows


def mean(rows, k):
    vals = [r[k] for r in rows if r.get(k) == r.get(k)]
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

    model = build_dit(size, input_size=input_size, patch_size=patch, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if args.tile <= 0:
        args.tile = default_tile
    print(
        f"loaded {ckpt_path} mode={mode} objective={objective} in_ch={in_ch} size={size} "
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

    def make_sr_fn(steps):
        if mode == "latent":
            return lambda lr_u8, sc: sr_latent_tiled(
                model, vae, vinfo, lr_u8, sc, steps, args.tile, args.pad, device, objective, args.seed,
                tile_batch=oom_state["tb"],
            )
        return lambda lr_u8, sc: sr_pixel_tiled(
            model, lr_u8, sc, steps, args.tile, args.pad, residual, device, objective, args.seed,
            tile_batch=oom_state["tb"],
        )

    sweep = []
    if args.sweep_steps and objective != "reg":
        sub = pairs[:: max(1, len(pairs) // max(args.sweep_pairs, 1))][: args.sweep_pairs]
        for s in [int(x) for x in args.sweep_steps.split(",") if x.strip()]:
            rows = run_pairs(sub, make_sr_fn(s), args.shave, oom_state)
            rec = {"steps": s, "nfe": 2 * s, "n": len(rows), "psnr_y": mean(rows, "psnr_y"), "ssim_y": mean(rows, "ssim_y")}
            sweep.append(rec)
            print(f"SWEEP steps={s} nfe={2*s} n={rec['n']} Y {rec['psnr_y']:.3f}/{rec['ssim_y']:.4f}", flush=True)

    rows = run_pairs(pairs, make_sr_fn(args.steps), args.shave, oom_state)

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
        "seed": args.seed,
        "sweep": sweep,
        "per_image": rows,
    }
    path = out_dir / "eval.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n== OFFICIAL n={summary['n']} mode={mode}/{objective} steps={args.steps} | "
        f"Y {summary['psnr_y']:.4f}/{summary['ssim_y']:.4f} | RGB {summary['psnr_rgb']:.2f} ==",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
