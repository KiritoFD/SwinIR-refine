"""Comprehensive IQA perceptual scoring suite via pyiqa.

Metrics supported:
  NR (No-Reference):
    - MUSIQ   -- Multi-scale Image Quality Transformer (Google, 2021) [higher is better]
    - MANIQA  -- Multi-dimension Attention Network (NTIRE'22 winner)  [higher is better]
    - NIQE    -- Natural Image Quality Evaluator (classical blind SR) [lower is better]
  FR (Full-Reference):
    - LPIPS   -- Learned Perceptual Image Patch Similarity (AlexNet)  [lower is better]
    - DISTS   -- Deep Image Structure and Texture Similarity          [lower is better]

Environment setup:
  HF_ENDPOINT=https://hf-mirror.com is automatically ensured for offline/restricted clusters.
"""

from __future__ import annotations

import os

# Ensure HF mirror is used if in restricted network
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

DEFAULT_METRICS = ("musiq", "maniqa", "niqe", "lpips", "dists")


class IQAScorer:
    """Lazily builds the pyiqa metrics on first use, then scores uint8 images."""

    def __init__(self, device: str | torch.device = "cuda", names=DEFAULT_METRICS):
        self.device = torch.device(device)
        self.names = tuple(names)
        self._metrics = None
        self._modes = {}

    def _build(self):
        if self._metrics is not None:
            return
        import pyiqa

        self._metrics = {}
        for n in self.names:
            try:
                m = pyiqa.create_metric(n, device=self.device)
                self._metrics[n] = m
                self._modes[n] = getattr(m, "metric_mode", "NR")
            except Exception as e:  # noqa: BLE001
                print(f"  [iqa] {n} unavailable: {type(e).__name__}: {str(e)[:120]}", flush=True)
        print(f"  [iqa] loaded {sorted(self._metrics)} on {self.device}", flush=True)

    def available(self) -> list[str]:
        self._build()
        return sorted(self._metrics)

    @torch.no_grad()
    def score(self, sr_u8, hr_u8=None) -> dict:
        """sr_u8: (H, W, 3) uint8. hr_u8: optional (H, W, 3) uint8 for FR metrics.
        Returns {metric_name: float}.
        """
        self._build()
        if not self._metrics:
            return {}
        sr_t = (
            torch.from_numpy(sr_u8)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        hr_t = None
        if hr_u8 is not None:
            hr_t = (
                torch.from_numpy(hr_u8)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )

        out = {}
        for n, m in self._metrics.items():
            mode = self._modes.get(n, "NR")
            try:
                if mode == "FR":
                    if hr_t is not None:
                        val = m(sr_t, hr_t)
                        out[n] = float(val.item() if hasattr(val, "item") else val)
                else:
                    val = m(sr_t)
                    out[n] = float(val.item() if hasattr(val, "item") else val)
            except Exception as e:  # noqa: BLE001
                print(f"  [iqa] {n} failed on image: {str(e)[:80]}", flush=True)
                out[n] = float("nan")
        return out
