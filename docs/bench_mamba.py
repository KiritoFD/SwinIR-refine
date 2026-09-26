"""Mamba (s1 d96) MFU at the largest batch that fits — its huge stride-1 activation
footprint is itself part of the finding. Tries batches 8/4/2/1."""
import time, gc, torch
from torch.utils.flop_counter import FlopCounterMode
from diffusion.mamba_sr import build_mambasr

dev, peak, HR = "cuda", 165.2e12, 128


def step(model, x, t, opt):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, t).float().pow(2).mean()
    loss.backward()
    opt.step()


def fwd_only(m, x, t):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(x, t).float().pow(2).mean()


ran = False
for bs in (8, 4, 2, 1):
    try:
        m = build_mambasr("S", input_size=HR, in_channels=3, dim=96, num_res=4,
                          backend="cuda", shift=False).to(dev)
        x = torch.randn(bs, 3, HR, HR, device=dev); t = torch.zeros(bs, device=dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        with FlopCounterMode(display=False) as fc:
            step(m, x, t, opt)
        flops = fc.get_total_flops()
        for _ in range(5):
            step(m, x, t, opt)
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(15):
            step(m, x, t, opt)
        torch.cuda.synchronize(); sec = (time.perf_counter() - s) / 15
        print(f"Mamba s1 d96 FORWARD+BACKWARD bs={bs}  {flops/bs/1e9:.1f} GFLOP/img  "
              f"achieved={flops/sec/1e12:.1f} TFLOPS  MFU={100*flops/sec/peak:.1f}%  "
              f"peakmem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        ran = True
        break
    except torch.cuda.OutOfMemoryError:
        print(f"Mamba bs={bs} fwd+bwd: OOM (backward needs too much activation)")
        try:
            del m, opt
        except Exception:
            pass
        gc.collect(); torch.cuda.empty_cache()

if not ran:
    for bs in (16, 8, 4, 2):
        try:
            m = build_mambasr("S", input_size=HR, in_channels=3, dim=96, num_res=4,
                              backend="cuda", shift=False).to(dev).eval()
            x = torch.randn(bs, 3, HR, HR, device=dev); t = torch.zeros(bs, device=dev)
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                fwd_only(m, x, t)
            flops = fc.get_total_flops()
            for _ in range(5):
                with torch.no_grad():
                    fwd_only(m, x, t)
            torch.cuda.synchronize(); s = time.perf_counter()
            for _ in range(20):
                with torch.no_grad():
                    fwd_only(m, x, t)
            torch.cuda.synchronize(); sec = (time.perf_counter() - s) / 20
            print(f"Mamba s1 d96 FORWARD-ONLY bs={bs}  {flops/bs/1e9:.1f} GFLOP/img  "
                  f"achieved={flops/sec/1e12:.1f} TFLOPS  fwd-MFU={100*flops/sec/peak:.1f}%  "
                  f"peakmem={torch.cuda.max_memory_allocated()/1e9:.1f}GB  "
                  f"(training backward OOMs even at bs=1 -> the real infra cost)")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"Mamba bs={bs} fwd-only: OOM too")
            try:
                del m
            except Exception:
                pass
            gc.collect(); torch.cuda.empty_cache()
