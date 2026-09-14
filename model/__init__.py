"""Minimal ModSwinIR for RealSR V3: residual bicubic + proven trunk + optional Align loss."""

from .model import ModSwinIR, build_model
from .dataset import RealSRPairDataset, build_realsr_index
from .losses import OffsetAlignedLoss, psnr
from .metrics import official_pair_metrics, rgb_to_y_matlab

__all__ = [
    "ModSwinIR",
    "build_model",
    "RealSRPairDataset",
    "build_realsr_index",
    "OffsetAlignedLoss",
    "psnr",
    "official_pair_metrics",
    "rgb_to_y_matlab",
]
