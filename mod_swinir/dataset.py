"""RealSR V3 paired dataset."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_pairs(scale_dir: Path, scale: int):
    """Match *_LR{scale}.png with *_HR.png in the same folder."""
    pairs = []
    if not scale_dir.is_dir():
        return pairs
    for lr_path in sorted(scale_dir.iterdir()):
        if lr_path.suffix.lower() not in IMG_EXTS:
            continue
        name = lr_path.name
        tag = f"_LR{scale}"
        if tag not in name:
            continue
        hr_name = name.replace(tag, "_HR")
        hr_path = scale_dir / hr_name
        if hr_path.exists():
            pairs.append((str(lr_path), str(hr_path), scale))
    return pairs


def build_realsr_index(root, cameras=("Canon", "Nikon"), split="Train", scales=(2, 3, 4)):
    root = Path(root)
    items = []
    for cam in cameras:
        for sc in scales:
            items.extend(_list_pairs(root / cam / split / str(sc), sc))
    return items


class RealSRPairDataset(Dataset):
    def __init__(
        self,
        root,
        split="Train",
        cameras=("Canon", "Nikon"),
        scales=(2, 3, 4),
        lr_patch=64,
        augment=True,
        max_pairs=None,
    ):
        self.root = root
        self.lr_patch = lr_patch
        self.augment = augment and split.lower() == "train"
        self.pairs = build_realsr_index(root, cameras, split, scales)
        if max_pairs is not None and len(self.pairs) > max_pairs:
            # deterministic subset for smoke runs
            rng = random.Random(0)
            self.pairs = rng.sample(self.pairs, max_pairs)
        if not self.pairs:
            raise RuntimeError(f"No RealSR pairs found under {root} split={split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, scale = self.pairs[idx]
        ps = self.lr_patch

        with Image.open(lr_path) as li, Image.open(hr_path) as hi:
            li = li.convert("RGB")
            hi = hi.convert("RGB")
            lw, lh = li.size
            hw, hh = hi.size
            # usable LR region given HR size
            max_lw, max_lh = hw // scale, hh // scale
            lw, lh = min(lw, max_lw), min(lh, max_lh)
            if lw < ps or lh < ps:
                # fall back: load small crops padded later via expand
                box_lr = (0, 0, lw, lh)
                box_hr = (0, 0, lw * scale, lh * scale)
                lr = np.asarray(li.crop(box_lr), dtype=np.uint8)
                hr = np.asarray(hi.crop(box_hr), dtype=np.uint8)
                pad_h = max(0, ps - lr.shape[0])
                pad_w = max(0, ps - lr.shape[1])
                lr = np.pad(lr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                hr = np.pad(
                    hr,
                    ((0, pad_h * scale), (0, pad_w * scale), (0, 0)),
                    mode="reflect",
                )
                top, left = 0, 0
                h, w = lr.shape[:2]
            else:
                if self.augment:
                    top = random.randint(0, lh - ps)
                    left = random.randint(0, lw - ps)
                else:
                    top = max(0, (lh - ps) // 2)
                    left = max(0, (lw - ps) // 2)
                # PIL crop: (left, top, right, bottom)
                lr = np.asarray(li.crop((left, top, left + ps, top + ps)), dtype=np.uint8)
                hr = np.asarray(
                    hi.crop(
                        (
                            left * scale,
                            top * scale,
                            (left + ps) * scale,
                            (top + ps) * scale,
                        )
                    ),
                    dtype=np.uint8,
                )

        if self.augment:
            if random.random() < 0.5:
                lr = lr[:, ::-1].copy()
                hr = hr[:, ::-1].copy()
            if random.random() < 0.5:
                lr = lr[::-1, :].copy()
                hr = hr[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                lr = np.ascontiguousarray(np.rot90(lr, k))
                hr = np.ascontiguousarray(np.rot90(hr, k))

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).contiguous()
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).contiguous()
        return {"lr": lr_t, "hr": hr_t, "scale": scale, "lr_path": lr_path}


class FullImageEvalDataset(Dataset):
    """Full LR/HR pair for validation (no crop). May be heavy — optional."""

    def __init__(self, root, split="Test", cameras=("Canon",), scales=(2,), max_pairs=4, max_side=512):
        self.pairs = build_realsr_index(root, cameras, split, scales)[:max_pairs]
        self.max_side = max_side

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, scale = self.pairs[idx]
        lr = np.asarray(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
        hr = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0
        # center crop max_side on LR
        h, w = lr.shape[:2]
        ms = self.max_side
        if h > ms or w > ms:
            top = max(0, (h - ms) // 2)
            left = max(0, (w - ms) // 2)
            ph = min(ms, h)
            pw = min(ms, w)
            lr = lr[top : top + ph, left : left + pw]
            hr = hr[top * scale : (top + ph) * scale, left * scale : (left + pw) * scale]
        return {
            "lr": torch.from_numpy(lr).permute(2, 0, 1).contiguous(),
            "hr": torch.from_numpy(hr).permute(2, 0, 1).contiguous(),
            "scale": scale,
            "lr_path": lr_path,
        }
