import os
import sys

# Ensure HF mirror is used
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pyiqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
candidates = ["niqe", "brisque", "musiq", "maniqa", "lpips", "dists", "clipiqa"]

print(f"Testing pyiqa metrics on {device} with HF_ENDPOINT={os.environ['HF_ENDPOINT']}...")
for name in candidates:
    print(f"Testing {name}...", flush=True)
    try:
        m = pyiqa.create_metric(name, device=device)
        dummy1 = torch.rand(1, 3, 128, 128, device=device)
        dummy2 = torch.rand(1, 3, 128, 128, device=device)
        mode = getattr(m, "metric_mode", "NR")
        if mode == "FR":
            val = m(dummy1, dummy2)
        else:
            val = m(dummy1)
        score = float(val.item() if hasattr(val, "item") else val)
        print(f"  --> {name:10s}: OK (mode={mode}, sample_score={score:.4f})", flush=True)
    except Exception as e:
        print(f"  --> {name:10s}: FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
