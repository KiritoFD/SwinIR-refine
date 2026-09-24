"""Muon optimizer (from sr-scaling) + AdamW composite for RealSR training."""

from __future__ import annotations

from typing import Iterable

import torch


def zeropower_via_newtonschulz5(matrix: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz expects 2D tensor")
    orig_dtype = matrix.dtype
    work = torch.bfloat16 if matrix.is_cuda else torch.float32
    x = matrix.to(work)
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D/conv hidden weights (lr typically 1e-3 on SR)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        params = list(params)
        if not params:
            raise ValueError("Muon needs parameters")
        if any(p.ndim < 2 for p in params):
            raise ValueError("Muon expects matrix/conv weights only")
        super().__init__(
            params,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "nesterov": nesterov,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            nesterov = bool(group["nesterov"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                shape = update.shape
                matrix = update.view(update.size(0), -1) if update.ndim > 2 else update
                matrix = zeropower_via_newtonschulz5(matrix, steps=ns_steps)
                matrix = matrix * max(1.0, matrix.size(-2) / matrix.size(-1)) ** 0.5
                update = matrix.view(shape)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class CompositeOptimizer:
    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = tuple(o for o in optimizers if o is not None)
        if not self.optimizers:
            raise ValueError("empty optimizers")

    @property
    def param_groups(self):
        groups = []
        for o in self.optimizers:
            groups.extend(o.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {f"optimizer_{i}": o.state_dict() for i, o in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, o in enumerate(self.optimizers):
            o.load_state_dict(state_dict[f"optimizer_{i}"])


def build_optimizer(model: torch.nn.Module, *, name: str = "adamw", lr: float = 2e-4, weight_decay: float = 1e-4,
                    momentum: float = 0.95, ns_steps: int = 5, aux_lr: float | None = None):
    name = name.lower()
    if name in ("adam", "adamw"):
        cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        return cls(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    if name != "muon":
        raise ValueError(name)
    if aux_lr is None:
        aux_lr = lr
    muon_p, aux_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_p.append(p)
        else:
            aux_p.append(p)
    opt = Muon(muon_p, lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
    if aux_p:
        return CompositeOptimizer(opt, torch.optim.AdamW(aux_p, lr=aux_lr, betas=(0.9, 0.99), weight_decay=0.0))
    return opt
