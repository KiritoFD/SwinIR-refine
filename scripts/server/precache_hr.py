"""Decode the pretraining HR images once, into a flat uint8 blob per source.

Why: PNG decode on every __getitem__ caps the loader at a few hundred
samples/s, and the U-Net at batch 390 wants ~900.  RAM is not the constraint
here (125 GB total, ~15 GB of decoded pixels), so the cheapest fix is to decode
once and keep the pixels in a single memmappable file.

Layout, per source root:
    data/pretrain_decoded/<name>.u8    raw RGB uint8, images concatenated
    data/pretrain_decoded/manifest.json  {name: {blob, items:[{path,offset,w,h}]}}

Reading back is then
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    img  = blob[offset : offset + w*h*3].reshape(h, w, 3)

Workers write with os.pwrite straight into a preallocated file, so the decoded
pixels never travel back through IPC (which would mean copying ~15 GB).

CPU only -- safe to run alongside GPU training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def find_images(root: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return out


def probe(path: str) -> tuple[int, int]:
    """Header-only read -- no pixel decode."""
    with Image.open(path) as im:
        return im.size  # (w, h)


def _worker(job):
    blob_path, offset, path = job
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    fd = os.open(blob_path, os.O_WRONLY)
    try:
        os.pwrite(fd, arr.tobytes(), offset)
    finally:
        os.close(fd)
    return arr.shape


def build(name: str, root: str, out_dir: Path, workers: int, verify: int) -> dict:
    files = find_images(root)
    if not files:
        print(f"  {name}: no images under {root}", flush=True)
        return {}
    print(f"  {name}: {len(files)} images, probing sizes ...", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        sizes = pool.map(probe, files, chunksize=64)

    offsets, off = [], 0
    for (w, h) in sizes:
        offsets.append(off)
        off += w * h * 3
    total = off
    print(f"  {name}: {total / 1024**3:.2f} GB decoded, probing took {time.time()-t0:.1f}s", flush=True)

    blob = out_dir / f"{name}.u8"
    with open(blob, "wb") as f:
        f.truncate(total)

    t0 = time.time()
    jobs = [(str(blob), offsets[i], files[i]) for i in range(len(files))]
    done = 0
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(_worker, jobs, chunksize=8):
            done += 1
            if done % 500 == 0 or done == len(files):
                el = time.time() - t0
                print(f"    {name}: {done}/{len(files)}  {el:.0f}s  "
                      f"({done/el:.0f} img/s)", flush=True)

    items = [
        {"path": files[i], "offset": offsets[i], "w": sizes[i][0], "h": sizes[i][1]}
        for i in range(len(files))
    ]

    if verify:
        mm = np.memmap(blob, dtype=np.uint8, mode="r")
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(items), size=min(verify, len(items)), replace=False)
        bad = 0
        for i in idxs:
            it = items[int(i)]
            got = mm[it["offset"] : it["offset"] + it["w"] * it["h"] * 3].reshape(
                it["h"], it["w"], 3
            )
            with Image.open(it["path"]) as im:
                ref = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(got, ref):
                bad += 1
        print(f"  {name}: verify {len(idxs)} random images -> {'OK' if bad == 0 else f'{bad} MISMATCH'}", flush=True)
        del mm

    return {"blob": str(blob), "n": len(items), "bytes": total, "items": items}


DEFAULT_ROOTS = ",".join(
    [
        "data/pretrain/DIV2K_train_HR",
        "data/pretrain/DIV2K_valid_HR",
        "data/pretrain/Flickr2K",
        "data/RealSR(V3)/Canon/Train/2",
        "data/RealSR(V3)/Nikon/Train/2",
        "data/RealSR(V3)/Canon/Test/2",
        "data/RealSR(V3)/Nikon/Test/2",
    ]
)


def slug(root: str) -> str:
    """Stable, collision-free key.  Several RealSR dirs are all called '2', so
    the leaf name is not enough -- use the sanitised whole path.  The key is
    cosmetic anyway: lookups go by file path, not by key."""
    return re.sub(r"[^a-z0-9]+", "_", str(root).lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default="/home/ds/realsr/data/decoded")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    t0 = time.time()
    for root in [r.strip() for r in args.roots.split(",") if r.strip()]:
        name = slug(root)
        if name in manifest and Path(manifest[name]["blob"]).exists():
            print(f"  {name}: already cached ({manifest[name]['n']} images), skipping", flush=True)
            continue
        rec = build(name, root, out_dir, args.workers, args.verify)
        if rec:
            manifest[name] = rec
            manifest_path.write_text(json.dumps(manifest))
            print(f"  {name}: manifest updated", flush=True)

    tot = sum(v["bytes"] for v in manifest.values()) / 1024**3
    n = sum(v["n"] for v in manifest.values())
    print(f"\nDONE  {n} images, {tot:.2f} GB on disk, {time.time()-t0:.0f}s")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
