"""VAE wrappers for RealSR diffusion (SD f8 / f4, Flux/SD3-style 16ch, custom path)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEInfo:
    name: str
    downscale: int
    latent_channels: int
    scaling_factor: float
    shift_factor: float
    source: str

    @property
    def norm_tag(self) -> str:
        return f"x{self.downscale}-c{self.latent_channels}-s{self.scaling_factor:g}-b{self.shift_factor:g}"


# Built-in presets. HF ids download on first use if network allows.
PRESETS: dict[str, dict] = {
    # Stable Diffusion 1.5 family
    "sd-vae-ft-ema": {
        "hf": "stabilityai/sd-vae-ft-ema",
        "downscale": 8,
        "latent_channels": 4,
        "scaling_factor": 0.18215,
    },
    "sd-vae-ft-mse": {
        "hf": "stabilityai/sd-vae-ft-mse",
        "downscale": 8,
        "latent_channels": 4,
        "scaling_factor": 0.18215,
    },
    # f4 KL — better recon, 3x latent tokens
    "kl-f4": {
        "hf": "madebyollin/sdxl-vae-fp16-fix",  # fallback; prefer local kl-f4 path
        "downscale": 4,
        "latent_channels": 4,
        "scaling_factor": 0.1268,  # approximate; override via --scaling-factor if local
        "note": "Use local path if you have Stability kl-f4 weights",
    },
    "sdxl-vae": {
        "hf": "madebyollin/sdxl-vae-fp16-fix",
        "downscale": 8,
        "latent_channels": 4,
        "scaling_factor": 0.13025,
    },
    # Public FLUX VAE mirrors (BFL repos often gated; these are open)
    # Flux latents are NOT zero-mean: z_norm = (z - 0.1159) * 0.3611
    "flux1-vae": {
        "hf": "diffusers/FLUX.1-vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    },
    "flux2-vae": {
        "hf": "unsloth/FLUX.2-VAE",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    },
    # FLUX.1 / SD3 lineage: f8, 16 latent channels
    "flux1-dev": {
        "hf": "black-forest-labs/FLUX.1-dev",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    },
    "flux1-schnell": {
        "hf": "black-forest-labs/FLUX.1-schnell",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    },
    # SD3 medium VAE (same family as Flux, often easier to pull)
    "sd3-vae": {
        "hf": "stabilityai/stable-diffusion-3-medium-diffusers",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 1.5305,
        "shift_factor": 0.0,
    },
}


def load_vae(
    name_or_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Load AutoencoderKL from preset name or local/HF path.

    Returns (vae, VAEInfo). VAE is in eval mode, no grad.
    Numbers that actually matter (downscale / latent_channels / scaling_factor /
    shift_factor) are taken from the model's own ``config.json`` whenever it
    provides them; the preset table is only a fallback for offline guesses.
    """
    from diffusers import AutoencoderKL

    name_or_path = str(name_or_path)
    if name_or_path in PRESETS:
        cfg = PRESETS[name_or_path]
        src = cfg["hf"]
        sub = cfg.get("subfolder")
        kwargs = {"subfolder": sub} if sub else {}
        try:
            vae = AutoencoderKL.from_pretrained(src, torch_dtype=dtype, **kwargs)
        except Exception as e:  # noqa: BLE001
            # offline / gated: retry without subfolder or with local cache hint
            raise RuntimeError(
                f"Failed to load VAE preset '{name_or_path}' from {src}: {e}\n"
                f"Pass a local directory to --vae if weights are already on disk."
            ) from e
        c = vae.config
        info = VAEInfo(
            name=name_or_path,
            downscale=int(cfg["downscale"]),
            latent_channels=int(cfg["latent_channels"]),
            scaling_factor=float(cfg["scaling_factor"]),
            shift_factor=float(cfg.get("shift_factor", 0.0)),
            source=src,
        )
    else:
        vae = AutoencoderKL.from_pretrained(name_or_path, torch_dtype=dtype)
        c = vae.config
        info = VAEInfo(
            name=name_or_path,
            downscale=int(2 ** (len(c.block_out_channels) - 1)),
            latent_channels=int(c.latent_channels),
            scaling_factor=float(getattr(c, "scaling_factor", 0.18215)),
            shift_factor=float(getattr(c, "shift_factor", 0.0)),
            source=name_or_path,
        )

    # config.json wins over the preset table (it is what the weights were trained with)
    if getattr(c, "latent_channels", None) is not None:
        info.latent_channels = int(c.latent_channels)
    if getattr(c, "block_out_channels", None):
        info.downscale = int(2 ** (len(c.block_out_channels) - 1))
    if getattr(c, "scaling_factor", None) is not None:
        info.scaling_factor = float(c.scaling_factor)
    if getattr(c, "shift_factor", None) is not None:
        info.shift_factor = float(c.shift_factor)

    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, info


@torch.no_grad()
def encode(vae, x: torch.Tensor, info: VAEInfo, sample: bool = False) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1] → normalized latent (z - shift) * scale.

    Normalized latent is ~(0,1) for SD/SDXL and ~N(0, ~1) for Flux once the
    0.1159 shift is removed — this is what makes flow matching well conditioned.
    """
    dtype = next(vae.parameters()).dtype
    x = x.to(dtype) * 2.0 - 1.0
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() if sample else posterior.mode()
    # always hand back fp32: the VAE may be bf16 for speed, the model/loss is not
    return ((z - info.shift_factor) * info.scaling_factor).float()


@torch.no_grad()
def decode(vae, z: torch.Tensor, info: VAEInfo) -> torch.Tensor:
    """normalized latent → (B,3,H,W) fp32 in [0,1]."""
    dtype = next(vae.parameters()).dtype
    z = (z / info.scaling_factor + info.shift_factor).to(dtype)
    x = vae.decode(z).sample
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


def decode_grad(vae, z: torch.Tensor, info: VAEInfo) -> torch.Tensor:
    """Same as decode() but differentiable w.r.t. z (VAE params stay frozen).

    Used only when --pixel-loss-weight > 0. Costs VAE-decoder activations.
    """
    dtype = next(vae.parameters()).dtype
    z = (z / info.scaling_factor + info.shift_factor).to(dtype)
    x = vae.decode(z).sample
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def ssim01(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap global SSIM on (B,3,H,W) in [0,1], mean over batch. Window 11 gaussian."""
    import math

    device = a.device
    c1 = 0.01**2
    c2 = 0.03**2
    size, sigma = 11, 1.5
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    win = torch.outer(g, g).view(1, 1, size, size)
    pad = size // 2

    def conv(x):
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), win)

    # mean over channels as gray
    ga = a.mean(1, keepdim=True)
    gb = b.mean(1, keepdim=True)
    mu_a, mu_b = conv(ga), conv(gb)
    sa = conv(ga * ga) - mu_a * mu_a
    sb = conv(gb * gb) - mu_b * mu_b
    sab = conv(ga * gb) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sa + sb + c2) + 1e-12
    )
    return float(ssim_map.mean().item())
