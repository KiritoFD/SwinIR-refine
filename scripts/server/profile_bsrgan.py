"""Where do BSRGAN's ~150 ms/item go?  Every op runs on a 128x128 patch, so
nothing here should cost more than a millisecond -- if JPEG dominates, the
cheapest fix is to make it optional."""
import random
import sys
import time

sys.path.insert(0, "/home/ds/realsr")

import torch
import torch.nn.functional as F

from diffusion import bsrgan as B

hr = torch.rand(3, 128, 128)
rng = random.Random(0)


def bench(fn, n=200):
    fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


print("threads:", torch.get_num_threads(), flush=True)
print(f"_blur            {bench(lambda: B._blur(hr, 2, rng)):8.2f} ms")
print(f"_resize          {bench(lambda: B._resize(hr, rng)):8.2f} ms")
print(f"_noise           {bench(lambda: B._noise(hr, rng)):8.2f} ms")
print(f"_jpeg            {bench(lambda: B._jpeg(hr, rng)):8.2f} ms")
k = B.sinc_kernel(1.0, 11)
print(f"sinc+conv        {bench(lambda: B._conv2d_sym(hr, k)):8.2f} ms")
print(f"interp bicubic   {bench(lambda: F.interpolate(hr[None], size=(64, 64), mode='bicubic', align_corners=False)[0]):8.2f} ms")
print(f"clone            {bench(lambda: hr.clone()):8.2f} ms")
print(f"FULL degrade     {bench(lambda: B.bsrgan_degrade(hr, 2)):8.2f} ms")
print(f"FULL (seed=0)    {bench(lambda: B.bsrgan_degrade(hr, 2, seed=0)):8.2f} ms")
