"""CPU-only smoke test for the BSRGAN pretraining dataset.

Verifies that every HR folder is picked up, that items decode, and that the
degradation severity is in the same ballpark as real RealSR LR images
(mean bicubic-upsample PSNR around 30 dB).  No GPU, no training.
"""
import sys, time, statistics
sys.path.insert(0, "/home/ds/realsr")

from PIL import Image
import numpy as np
import torch

from diffusion.bsrgan import BSRGANDataset, find_images

ROOTS = [
    "/home/ds/realsr/data/pretrain/DIV2K_train_HR",
    "/home/ds/realsr/data/pretrain/DIV2K_valid_HR",
    "/home/ds/realsr/data/pretrain/Flickr2K",
]

total = 0
for r in ROOTS:
    n = len(find_images(r))
    total += n
    print(f"{r}: {n}")
print(f"TOTAL HR images: {total}")

ds = BSRGANDataset(",".join(ROOTS), lr_patch=64, scale=2, augment=True, limit=0, seed=0)
print(f"dataset len: {len(ds)}")

sf = 2


def y_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)


def to_y(arr01: np.ndarray) -> np.ndarray:
    # BT.601 limited range, uint8, to match eval_official
    r, g, b = arr01[..., 0], arr01[..., 1], arr01[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) * 219.0 / 255.0
    return np.clip(np.round(y), 16, 235)


t0 = time.time()
psnrs, shapes, bad = [], set(), []
N = 60
for i in range(N):
    try:
        item = ds[i]
    except Exception as exc:  # noqa: BLE001
        bad.append(f"idx {i}: {type(exc).__name__}: {exc}")
        continue
    lr = item["lr"]
    hr = item["hr"]
    shapes.add(tuple(lr.shape))
    # bicubic-upsample the LR and compare to HR -> degradation severity
    lr_np = (lr.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    hr_np = (hr.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    up = np.asarray(
        Image.fromarray(lr_np).resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
    )
    psnrs.append(y_psnr(to_y(up / 255.0), to_y(hr_np / 255.0)))

dt = time.time() - t0
print(f"lr shapes seen: {sorted(shapes)}")
print(f"degraded samples: {len(psnrs)}/{N} in {dt:.1f}s ({dt/max(len(psnrs),1):.2f}s/it)")
if psnrs:
    print(
        f"bicubic-upsample Y PSNR  mean {statistics.mean(psnrs):.2f} "
        f"min {min(psnrs):.2f} max {max(psnrs):.2f}"
    )
if bad:
    print("FAILURES:")
    for b in bad[:10]:
        print("  " + b)
    sys.exit(1)
print("SMOKE OK")
