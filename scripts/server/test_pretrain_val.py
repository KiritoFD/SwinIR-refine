"""Sanity-check the new pretrain val: deterministic, disjoint from train."""
import sys

sys.path.insert(0, "/home/ds/realsr")

import torch

from diffusion.bsrgan import BSRGANDataset

M = "data/decoded/manifest.json"
tr = BSRGANDataset("data/pretrain/DIV2K_train_HR,data/pretrain/Flickr2K",
                   64, 2, True, 0, 42, decoded_manifest=M)
va = BSRGANDataset("data/pretrain/DIV2K_valid_HR",
                   64, 2, False, 0, 42, decoded_manifest=M, deterministic=True)

print(f"train {len(tr)} imgs / val {len(va)} imgs")

a = va[0]["lr"]
b = va[0]["lr"]
c = va[1]["lr"]
print("val is deterministic across calls :", bool(torch.equal(a, b)))
print("different index gives different data:", not bool(torch.equal(a, c)))

overlap = set(tr.files) & set(va.files)
print("train/val file overlap:", len(overlap))

# and the train set must still be stochastic
t1, t2 = tr[0]["lr"], tr[0]["lr"]
print("train is stochastic             :", not bool(torch.equal(t1, t2)))
