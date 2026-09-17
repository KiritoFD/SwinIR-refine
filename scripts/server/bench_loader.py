"""Measure real DataLoader throughput (with workers) for both datasets."""
import sys
import time

sys.path.insert(0, "/home/ds/realsr")

import torch
from torch.utils.data import DataLoader

from diffusion.bsrgan import BSRGANDataset
from diffusion.data import RealSRCropDataset
from diffusion.train_pixel import _worker_init

M = "data/decoded/manifest.json"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def run(name, ds, batch, n=20):
    loader = DataLoader(
        ds, batch_size=batch, shuffle=True, num_workers=W, pin_memory=True,
        drop_last=True, persistent_workers=True, worker_init_fn=_worker_init,
    )
    it = iter(loader)
    for _ in range(3):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)
    t0 = time.time()
    got = 0
    for _ in range(n):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        got += b["lr"].shape[0]
    dt = time.time() - t0
    print(f"  {name:14s} workers={W} batch={batch}  {dt/n*1000:7.1f} ms/batch  "
          f"-> {got/dt:7.1f} samples/s", flush=True)


print("threads in main:", torch.get_num_threads(), flush=True)
pt = BSRGANDataset(
    "data/pretrain/DIV2K_train_HR,data/pretrain/DIV2K_valid_HR,data/pretrain/Flickr2K",
    64, 2, True, 0, 42, decoded_manifest=M,
)
rs = RealSRCropDataset("data/RealSR(V3)", "Train", ("Canon", "Nikon"), 2, 64, True,
                       decoded_manifest=M)

which = sys.argv[2] if len(sys.argv) > 2 else "both"
if which in ("both", "bsrgan"):
    run("BSRGAN", pt, 768)
if which in ("both", "realsr"):
    run("RealSR", rs, 390)
