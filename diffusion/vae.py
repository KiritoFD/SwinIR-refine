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
    source: str


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
    "flux1-vae": {
        "hf": "diffusers/FLUX.1-vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
    },
    "flux2-vae": {
        "hf": "unsloth/FLUX.2-VAE",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
    },
    # FLUX.1 / SD3 lineage: f8, 16 latent channels
    "flux1-dev": {
        "hf": "black-forest-labs/FLUX.1-dev",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
    },
    "flux1-schnell": {
        "hf": "black-forest-labs/FLUX.1-schnell",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 0.3611,
    },
    # SD3 medium VAE (same family as Flux, often easier to pull)
    "sd3-vae": {
        "hf": "stabilityai/stable-diffusion-3-medium-diffusers",
        "subfolder": "vae",
        "downscale": 8,
        "latent_channels": 16,
        "scaling_factor": 1.5305,
    },
}


def load_vae(
    name_or_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Load AutoencoderKL from preset name or local/HF path.

    Returns (vae, VAEInfo). VAE is in eval mode, no grad.
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
        info = VAEInfo(
            name=name_or_path,
            downscale=int(cfg["downscale"]),
            latent_channels=int(cfg["latent_channels"]),
            scaling_factor=float(cfg["scaling_factor"]),
            source=src,
        )
    else:
        vae = AutoencoderKL.from_pretrained(name_or_path, torch_dtype=dtype)
        ds = 2 ** (len(vae.config.block_out_channels) - 1)
        lc = int(vae.config.latent_channels)
        sf = float(getattr(vae.config, "scaling_factor", 0.18215))
        info = VAEInfo(
            name=name_or_path,
            downscale=int(ds),
            latent_channels=lc,
            scaling_factor=sf,
            source=name_or_path,
        )

    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, info


@torch.no_grad()
def encode(vae, x: torch.Tensor, info: VAEInfo, sample: bool = False) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1] → latent (B,C,h,w), already * scaling_factor."""
    x = x * 2.0 - 1.0
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() if sample else posterior.mode()
    return z * info.scaling_factor


@torch.no_grad()
def decode(vae, z: torch.Tensor, info: VAEInfo) -> torch.Tensor:
    """latent → (B,3,H,W) in [0,1]."""
    z = z / info.scaling_factor
    x = vae.decode(z).sample
    return ((x + 1.0) * 0.5).clamp(0, 1)


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
