"""RealSR V3 paired crops for diffusion training/eval."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_pairs(scale_dir: Path, scale: int):
    pairs = []
    if not Path(scale_dir).is_dir():
        return pairs
    for lr_path in sorted(Path(scale_dir).iterdir()):
        if lr_path.suffix.lower() not in IMG_EXTS:
            continue
        tag = f"_LR{scale}"
        if tag not in lr_path.name:
            continue
        hr_path = lr_path.with_name(lr_path.name.replace(tag, "_HR"))
        if hr_path.exists():
            pairs.append((str(lr_path), str(hr_path), scale))
    return pairs


def build_index(root, cameras=("Canon", "Nikon"), split="Test", scale=2):
    root = Path(root)
    items = []
    for cam in cameras:
        items.extend(list_pairs(root / cam / split / str(scale), scale))
    return items


def _to_rgb_u8(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


class RealSRCropDataset(Dataset):
    """Random LR crop of size lr_patch; HR crop is scale× larger. RGB float in [0,1]."""

    def __init__(
        self,
        root: str,
        split: str = "Train",
        cameras=("Canon", "Nikon"),
        scale: int = 2,
        lr_patch: int = 64,
        augment: bool = True,
        max_pairs: int | None = None,
    ):
        self.scale = scale
        self.lr_patch = lr_patch
        self.augment = augment and split.lower() == "train"
        self.pairs = build_index(root, cameras, split, scale)
        if not self.pairs:
            raise RuntimeError(f"No pairs under {root} {split} x{scale}")
        if max_pairs is not None and len(self.pairs) > max_pairs:
            rng = random.Random(0)
            self.pairs = rng.sample(self.pairs, max_pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_path, hr_path, sc = self.pairs[idx]
        ps = self.lr_patch
        with Image.open(lr_path) as li, Image.open(hr_path) as hi:
            lw, lh = li.size
            hw, hh = hi.size
            max_lw, max_lh = hw // sc, hh // sc
            lw, lh = min(lw, max_lw), min(lh, max_lh)
            if lw < ps or lh < ps:
                box_lr = (0, 0, max(lw, 1), max(lh, 1))
                box_hr = (0, 0, max(lw, 1) * sc, max(lh, 1) * sc)
                lr = np.asarray(li.crop(box_lr).convert("RGB"), dtype=np.uint8)
                hr = np.asarray(hi.crop(box_hr).convert("RGB"), dtype=np.uint8)
                pad_h = max(0, ps - lr.shape[0])
                pad_w = max(0, ps - lr.shape[1])
                if pad_h or pad_w:
                    lr = np.pad(lr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                    hr = np.pad(
                        hr,
                        ((0, pad_h * sc), (0, pad_w * sc), (0, 0)),
                        mode="reflect",
                    )
            else:
                if self.augment:
                    top = random.randint(0, lh - ps)
                    left = random.randint(0, lw - ps)
                else:
                    top = (lh - ps) // 2
                    left = (lw - ps) // 2
                lr = np.asarray(
                    li.crop((left, top, left + ps, top + ps)).convert("RGB"),
                    dtype=np.uint8,
                )
                hr = np.asarray(
                    hi.crop(
                        (left * sc, top * sc, (left + ps) * sc, (top + ps) * sc)
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
        if self.augment:
            if random.random() < 0.5:
                lr = lr[:, ::-1].copy()
                hr = hr[:, ::-1].copy()
            if random.random() < 0.5:
                lr = lr[::-1].copy()
                hr = hr[::-1].copy()
            k = random.randint(0, 3)
            if k:
                lr = np.ascontiguousarray(np.rot90(lr, k))
                hr = np.ascontiguousarray(np.rot90(hr, k))

        def to_t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0

        return {"lr": to_t(lr), "hr": to_t(hr), "scale": sc}


class RealSRFullEvalDataset(Dataset):
    """Full Test pairs (no crop) for VAE noise floor / tiled SR eval."""

    def __init__(self, root: str, cameras=("Canon", "Nikon"), scale: int = 2, max_pairs: int | None = None):
        self.pairs = build_index(root, cameras, "Test", scale)
        if max_pairs is not None and len(self.pairs) > max_pairs:
            step = max(1, len(self.pairs) // max_pairs)
            self.pairs = self.pairs[::step][:max_pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_path, hr_path, sc = self.pairs[idx]
        return {
            "lr": _to_rgb_u8(lr_path),
            "hr": _to_rgb_u8(hr_path),
            "scale": sc,
            "name": Path(lr_path).name,
        }
