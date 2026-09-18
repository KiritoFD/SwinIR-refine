"""RealSR V3 paired crops for diffusion training/eval."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_CANDIDATE_ROOTS = (
    r"G:\RealSR\data\RealSR(V3)",
    "/home/ds/realsr/data/RealSR(V3)",
    "data/RealSR(V3)",
)


def default_root() -> str:
    """Pick a RealSR root that actually exists (Windows box vs dserver)."""
    import os

    env = os.environ.get("REALSR_DATA", "")
    for c in (env, *_CANDIDATE_ROOTS):
        if c and Path(c).is_dir():
            return c
    return _CANDIDATE_ROOTS[0]


def make_split(ds, n_val: int, seed: int = 0):
    """Deterministic train/val split by pair index. val copy gets augment=False."""
    import copy as _copy
    import random as _random

    if n_val <= 0:
        return ds, None
    idx = list(range(len(ds.pairs)))
    _random.Random(seed).shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = _copy.copy(ds)
    train.pairs = [ds.pairs[i] for i in train_idx]
    val = _copy.copy(ds)
    val.pairs = [ds.pairs[i] for i in val_idx]
    val.augment = False
    return train, val


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
        cache: bool = False,
        decoded_manifest: str | None = None,
    ):
        from .decoded import DecodedStore

        self.scale = scale
        self.lr_patch = lr_patch
        self.augment = augment and split.lower() == "train"
        self.pairs = build_index(root, cameras, split, scale)
        if not self.pairs:
            raise RuntimeError(f"No pairs under {root} {split} x{scale}")
        if max_pairs is not None and len(self.pairs) > max_pairs:
            rng = random.Random(0)
            self.pairs = rng.sample(self.pairs, max_pairs)
        self.store = DecodedStore(decoded_manifest)
        if self.store:
            hits = sum(1 for lp, hp, _ in self.pairs if self.store.has(lp) and self.store.has(hp))
            print(f"  RealSRCropDataset: {hits}/{len(self.pairs)} pairs from the decoded cache",
                  flush=True)
            if hits == 0:
                print(f"  WARNING: the decoded cache has {self.store.n_images} images but none "
                      f"match this dataset's paths -- every item will decode from PNG. "
                      f"Check that --data-root matches how precache_hr.py was run.",
                      flush=True)

        # Decoding the PNGs on every __getitem__ caps the loader at a few hundred
        # samples/s -- far below what a 700-sample batch at 0.43 s/step needs.
        # Preloading here (not lazily) matters: DataLoader forks its workers AFTER
        # __init__, so the arrays are shared copy-on-write instead of being
        # duplicated per worker. 406 pairs is ~2 GB.
        self._cache = None
        if cache:
            self._cache = [self._load_full(i) for i in range(len(self.pairs))]

    def _load_full(self, idx: int):
        lr_path, hr_path, _sc = self.pairs[idx]
        with Image.open(lr_path) as li, Image.open(hr_path) as hi:
            lr = np.asarray(li.convert("RGB"), dtype=np.uint8)
            hr = np.asarray(hi.convert("RGB"), dtype=np.uint8)
        return lr, hr

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ps = self.lr_patch
        lr_path, hr_path, sc = self.pairs[idx]
        lr_full, hr_full = self.store.get(lr_path), self.store.get(hr_path)
        if lr_full is None or hr_full is None:
            lr_full, hr_full = self._cache[idx] if self._cache is not None else self._load_full(idx)
        lh, lw = lr_full.shape[:2]
        hh, hw = hr_full.shape[:2]
        lw, lh = min(lw, hw // sc), min(lh, hh // sc)
        if lw < ps or lh < ps:
            lr = lr_full[: max(lh, 1), : max(lw, 1)]
            hr = hr_full[: max(lh, 1) * sc, : max(lw, 1) * sc]
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
            lr = lr_full[top : top + ps, left : left + ps]
            hr = hr_full[top * sc : (top + ps) * sc, left * sc : (left + ps) * sc]
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


class LatentCropDataset(Dataset):
    """Random crops from PRE-ENCODED latents (see diffusion.precompute_latents).

    Encoding the VAE twice per step costs ~1.0 s/step at HR512 — more than the
    DiT itself. Pre-encoding once and cropping in latent space removes it, and is
    equivalent up to boundary padding because the VAE is a pure 8x conv stack.
    """

    def __init__(self, cache_dir: str, crop: int, augment: bool = True, max_pairs: int | None = None,
                 keep=None):
        self.crop = crop
        self.augment = augment
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if keep is not None:
            names = set(keep)
            self.files = [f for f in self.files if f.stem in names]
        if not self.files:
            raise RuntimeError(f"no cached latents in {cache_dir}")
        if max_pairs is not None and len(self.files) > max_pairs:
            self.files = self.files[:max_pairs]
        self._cache = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, i):
        if i not in self._cache:
            d = torch.load(self.files[i], map_location="cpu", weights_only=False)
            self._cache[i] = (d["z_hr"].float(), d["z_lr"].float())
        return self._cache[i]

    def __getitem__(self, idx: int):
        z0, zc = self._load(idx)
        c, h, w = z0.shape
        ch = cw = self.crop
        if h < ch or w < cw:
            pad_h, pad_w = max(0, ch - h), max(0, cw - w)
            z0 = torch.nn.functional.pad(z0, (0, pad_w, 0, pad_h))
            zc = torch.nn.functional.pad(zc, (0, pad_w, 0, pad_h))
            h, w = z0.shape[-2:]
        if self.augment:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
        else:
            top, left = (h - ch) // 2, (w - cw) // 2
        z0 = z0[:, top : top + ch, left : left + cw]
        zc = zc[:, top : top + ch, left : left + cw]
        if self.augment:
            if random.random() < 0.5:
                z0, zc = z0.flip(-1), zc.flip(-1)
            if random.random() < 0.5:
                z0, zc = z0.flip(-2), zc.flip(-2)
            k = random.randint(0, 3)
            if k:
                z0, zc = torch.rot90(z0, k, (-2, -1)), torch.rot90(zc, k, (-2, -1))
        return {"z0": z0, "zc": zc, "name": self.files[idx].stem}


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
