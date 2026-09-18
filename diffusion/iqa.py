"""MUSIQ / MANIQA no-reference perceptual scores, via pyiqa.

Why: PSNR and SSIM measure fidelity to the ground truth and are blind to how an
image actually *looks*.  A model that over-smooths gains PSNR and loses
perception, and nothing in our current table would notice.  These two are
NR-IQA models trained on human opinion scores, so they do.

    MUSIQ  -- Multi-scale Image Quality Transformer (Google, 2021)
    MANIQA -- Multi-dimension Attention Network, NTIRE 2022 winner

Weights: github.com is unreachable from the training box, but pyiqa also pulls
from the HF mirror `chaofengc/IQA-PyTorch-Weights`, so simply setting
HF_ENDPOINT=https://hf-mirror.com is enough -- it fetches musiq_koniq_ckpt and
ckpt_koniq10k automatically into ~/.cache/torch/hub/pyiqa/.

Do NOT `pip install pyiqa` without --no-deps: its dependency spec pulls a newer
torch and would replace the cu128 build this env is pinned to.

Scores are relative, not absolute: compare within a table, not against a paper.
"""

from __future__ import annotations

import torch

DEFAULT_METRICS = ("musiq", "maniqa")


class IQAScorer:
    """Lazily builds the pyiqa metrics on first use, then scores uint8 images."""

    def __init__(self, device: str | torch.device = "cuda", names=DEFAULT_METRICS):
        self.device = torch.device(device)
        self.names = tuple(names)
        self._metrics = None

    def _build(self):
        if self._metrics is not None:
            return
        import pyiqa

        self._metrics = {}
        for n in self.names:
            try:
                self._metrics[n] = pyiqa.create_metric(n, device=self.device)
            except Exception as e:  # noqa: BLE001
                print(f"  [iqa] {n} unavailable: {type(e).__name__}: {str(e)[:120]}", flush=True)
        print(f"  [iqa] loaded {sorted(self._metrics)} on {self.device}", flush=True)

    def available(self) -> list[str]:
        self._build()
        return sorted(self._metrics)

    @torch.no_grad()
    def score(self, sr_u8) -> dict:
        """sr_u8: (H, W, 3) uint8.  Returns {metric_name: float}."""
        self._build()
        if not self._metrics:
            return {}
        t = (
            torch.from_numpy(sr_u8)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        out = {}
        for n, m in self._metrics.items():
            try:
                out[n] = float(m(t).item())
            except Exception as e:  # noqa: BLE001
                print(f"  [iqa] {n} failed on this image: {str(e)[:80]}", flush=True)
                out[n] = float("nan")
        return out
