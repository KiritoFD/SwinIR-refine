"""Measure MambaSR MFU for BOTH scan backends, forward+backward:
 - mamba_ssm : official fused CUDA selective-scan kernel (what the Mamba matrix
               actually trained with — see logs `backend=mamba_ssm ... available=True`).
 - torch     : pure-PyTorch parallel-scan S6 fallback (fp32, 4-direction batch stack).
Uses torch FlopCounterMode (counts matmul/conv; the fused scan kernel's own flops are
under-counted, so mamba_ssm's achieved TFLOPS here is a LOWER bound on its true compute).
"""
import time, gc, torch
from torch.utils.flop_counter import FlopCounterMode
from diffusion.mamba_sr import build_mambasr, mamba_ssm_available

dev, peak, HR = "cuda", 165.2e12, 128
print("mamba_ssm available:", mamba_ssm_available())


def step(model, x, t, opt):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, t).float().pow(2).mean()
    loss.backward()
    opt.step()


def measure(backend, bs):
    m = build_mambasr("S", input_size=HR, in_channels=3, dim=96, num_groups=2,
                      num_res=3, d_state=16, expand=2.0, backend=backend).to(dev)
    x = torch.randn(bs, 3, HR, HR, device=dev); t = torch.zeros(bs, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    with FlopCounterMode(display=False) as fc:
        step(m, x, t, opt)
    flops = fc.get_total_flops()
    for _ in range(5):
        step(m, x, t, opt)
    torch.cuda.synchronize(); s = time.perf_counter()
    iters = max(5, int(64 // bs))
    for _ in range(iters):
        step(m, x, t, opt)
    torch.cuda.synchronize(); sec = (time.perf_counter() - s) / iters
    tf = flops / sec
    print(f"[{backend:9s}] bs={bs:>2d}  fwd+bwd={flops/bs/1e9:7.1f} GFLOP/img  "
          f"{sec/1:.3f}s/iter  achieved>={tf/1e12:6.1f} TFLOPS  "
          f"MFU>={100*tf/peak:5.1f}%  peakmem={torch.cuda.max_memory_allocated()/1e9:5.1f}GB")
    del m, opt, x, t
    gc.collect(); torch.cuda.empty_cache()


for backend in ("mamba_ssm", "torch"):
    for bs in (16, 8, 4, 2, 1):
        try:
            measure(backend, bs)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[{backend:9s}] bs={bs} OOM; halving")
            gc.collect(); torch.cuda.empty_cache()
