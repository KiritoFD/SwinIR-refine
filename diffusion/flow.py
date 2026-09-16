"""Rectified flow matching (SD3/Flux-style) with logit-normal t and Heun sampling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TIME_SCALE = 1000.0


def sample_t(n: int, device, mode: str = "logit_normal", mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(n, device=device)
    if mode == "logit_normal":
        z = torch.randn(n, device=device) * std + mean
        return torch.sigmoid(z)
    if mode == "cosmap":
        u = torch.rand(n, device=device)
        return 1.0 - 1.0 / (torch.tan(math.pi * u / 2.0) + 1.0)
    raise ValueError(mode)


def flow_loss(model, x0: torch.Tensor, cond: torch.Tensor | None, t: torch.Tensor | None = None, t_mode: str = "logit_normal"):
    """MSE on velocity v = noise - x0.

    If cond is not None, model input is cat([x_t, cond], dim=1) and loss is
    only on the first x0.shape[1] channels.
    """
    b = x0.shape[0]
    device = x0.device
    if t is None:
        t = sample_t(b, device, t_mode)
    noise = torch.randn_like(x0)
    t_ = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * noise
    v_target = noise - x0
    x_in = torch.cat([x_t, cond], dim=1) if cond is not None else x_t
    v_pred = model(x_in, t * TIME_SCALE)
    if cond is not None:
        v_pred = v_pred[:, : x0.shape[1]]
    return F.mse_loss(v_pred, v_target), {"t": t.detach(), "loss": F.mse_loss(v_pred, v_target).detach()}


@torch.no_grad()
def sample_flow(
    model,
    shape: tuple,
    cond: torch.Tensor | None = None,
    steps: int = 25,
    solver: str = "heun",
    shift: float = 1.0,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
    seed: int | None = None,
):
    """Integrate ODE from t=1 (noise) to t=0 (data). Returns x0.

    Pass ``seed`` for a reproducible run (the same seed gives the same initial
    noise every tile/step, which removes sampling variance from the PSNR number).
    """
    if generator is None and seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(shape, device=device, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if shift != 1.0:
        s = torch.linspace(1.0, 0.0, steps + 1, device=device)
        ts = shift * s / (1 + (shift - 1.0) * s)

    def vf(xx, t_scalar):
        t_batch = torch.full((xx.shape[0],), float(t_scalar), device=device)
        x_in = torch.cat([xx, cond], dim=1) if cond is not None else xx
        v = model(x_in, t_batch * TIME_SCALE)
        if cond is not None:
            v = v[:, : shape[1]]
        return v

    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        v0 = vf(x, t0)
        if solver == "euler":
            x = x + dt * v0
        else:
            x_euler = x + dt * v0
            v1 = vf(x_euler, t1)
            x = x + dt * 0.5 * (v0 + v1)
    return x
