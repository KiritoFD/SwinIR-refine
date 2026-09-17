"""Sanity-check the BSRGAN degradation + pretraining dataset."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ds/realsr")
from diffusion.bsrgan import BSRGANDataset, bsrgan_degrade  # noqa: E402

ROOT = Path("/home/ds/realsr/data/pretrain")


def ensure(z: Path, out: Path) -> Path | None:
    if not z.is_file():
        return None
    if not any(out.iterdir()) if out.is_dir() else True:
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    return out


def main() -> None:
    for name in ("DIV2K_valid_HR", "DIV2K_train_HR", "Flickr2K"):
        d = ensure(ROOT / f"{name}.zip", ROOT / name)
        if d is None:
            continue
        n = len([p for p in d.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not n:
            continue
        print(f"=== {name}: {n} images ===")
        ds = BSRGANDataset(str(d), lr_patch=64, scale=2)
        psnrs = []
        for i in range(min(8, len(ds))):
            b = ds[i]
            lr, hr = b["lr"], b["hr"]
            up = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
            mse = ((up - hr) ** 2).mean().item()
            psnr = 10 * np.log10(1 / max(mse, 1e-12))
            psnrs.append(psnr)
            print(f"  lr{tuple(lr.shape)} hr{tuple(hr.shape)} "
                  f"range[{lr.min():.2f},{lr.max():.2f}] bicubic {psnr:.2f} dB")
        print(f"  mean bicubic {np.mean(psnrs):.2f} dB  (RealSR real-degradation ref ~32-33)")

        # save one example for eyeballing
        b = ds[0]
        lr, hr = b["lr"], b["hr"]
        up = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
        from PIL import Image
        grid = torch.cat([up[0], hr[0]], dim=1).clamp(0, 1).mul(255).byte().cpu().numpy()
        Image.fromarray(grid).save(ROOT / f"sample_{name}.png")
        print(f"  wrote {ROOT / f'sample_{name}.png'}")


if __name__ == "__main__":
    main()
