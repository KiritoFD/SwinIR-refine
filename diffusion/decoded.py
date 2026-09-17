"""Read images straight out of the pre-decoded uint8 blobs.

Written by ``scripts/server/precache_hr.py``.  Decoding a PNG on every
``__getitem__`` caps a loader at a few hundred samples/s; the U-Net at batch 390
wants ~900 and the pretrain set wants more.  Since RAM is not scarce here
(125 GB), the fix is to decode once into one flat blob per source and mmap it.

The lookup key is the **original file path**, so any dataset can use the cache
without agreeing on a naming convention:

    store = DecodedStore("data/decoded/manifest.json")
    img = store.get("/abs/path/to/x.png")      # (H, W, 3) uint8, or None

``None`` means "not cached" and the caller should fall back to decoding.  The
returned array is a view into the memmap, so it is zero-copy; copy it before
in-place augmentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_MAPS: dict[str, np.memmap] = {}


def _blob(path: str) -> np.memmap:
    """One memmap per process per file.  Under fork the pages are shared."""
    m = _MAPS.get(path)
    if m is None:
        m = np.memmap(path, dtype=np.uint8, mode="r")
        _MAPS[path] = m
    return m


class DecodedStore:
    def __init__(self, manifest_path: str | Path | None):
        self.entries: dict = {}
        self.by_path: dict[str, tuple[str, int]] = {}
        if manifest_path:
            p = Path(manifest_path)
            if p.is_file():
                self.entries = json.loads(p.read_text())
                for name, e in self.entries.items():
                    for i, it in enumerate(e["items"]):
                        self.by_path[it["path"]] = (name, i)

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def n_images(self) -> int:
        return sum(e["n"] for e in self.entries.values())

    @property
    def gb(self) -> float:
        return sum(e["bytes"] for e in self.entries.values()) / 1024**3

    def has(self, path: str) -> bool:
        return str(path) in self.by_path

    def get(self, path: str) -> np.ndarray | None:
        """(H, W, 3) uint8 view into the blob, or None if this path is not cached."""
        hit = self.by_path.get(str(path))
        if hit is None:
            return None
        name, idx = hit
        e = self.entries[name]
        it = e["items"][idx]
        w, h = int(it["w"]), int(it["h"])
        off = int(it["offset"])
        return _blob(e["blob"])[off : off + w * h * 3].reshape(h, w, 3)
