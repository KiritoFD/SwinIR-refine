"""Independent FLOPs recount for MFU sanity: count every Conv2d/ConvTranspose/Linear
as 2*MACs (forward). No GPU needed. Prints GFLOPs/image forward for our backbones so we
can re-derive MFU = (fwd_GFLOPs * samples/s [or 3x for fwd+bwd]) / peak.
Usage: python docs/bench_mfu.py [--samples-per-sec 218]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn


def count_fwd_flops(model, x):
    total = [0]
    hooks = []

    def conv_hook(m, inp, out):
        # 2 * Cout*K^2*Cin_group* Hout*Wout  (output-element based, handles groups)
        out_t = out if isinstance(out, torch.Tensor) else out[0]
        b, c = out_t.shape[0], out_t.shape[1]
        hw = out_t.shape[2] * out_t.shape[3] if out_t.dim() > 2 else 1
        k = m.kernel_size[0] * m.kernel_size[1] if m.kernel_size else 1
        cout = m.out_channels
        cin_per_group = m.in_channels // m.groups
        total[0] += 2 * cout * cin_per_group * k * hw * b

    def lin_hook(m, inp, out):
        total[0] += 2 * m.in_features * m.out_features * out.shape[0]

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(x, torch.zeros(x.shape[0]))
    for h in hooks:
        h.remove()
    return total[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-sec", type=float, default=218.0,
                    help="measured training throughput for b64 (128 / s_per_step)")
    ap.add_argument("--peak", type=float, default=165.2, help="RTX4090 bf16 dense TFLOPS")
    a = ap.parse_args()
    dev = "cpu"
    from diffusion.unet import build_unet

    cases = {}
    x64 = torch.rand(1, 3, 128, 128)
    cases["unet b64 (stride-1, HR128)"] = (build_unet("S", input_size=128, in_channels=3, out_channels=3,
                                                       base=64, mult=(1, 2, 4, 4), num_res=2,
                                                       attn_levels=(2, 3), in_stride=1, out_scale=1).to(dev), x64)
    x128 = torch.rand(1, 3, 128, 128)
    cases["unet b128 (stride-1, HR128)"] = (build_unet("S", input_size=128, in_channels=3, out_channels=3,
                                                        base=128, mult=(1, 2, 4, 4), num_res=2,
                                                        attn_levels=(2, 3), in_stride=1, out_scale=1).to(dev), x128)
    # DiT-S: 12 layers, d=384, patch2, tokens=(128/2)^2=4096, in 3ch
    from diffusion.dit import build_dit
    cases["dit-S (HR128, patch2)"] = (build_dit("S", input_size=128, patch_size=2, in_channels=6).to(dev),
                                      torch.rand(1, 6, 128, 128))

    print(f"{'backbone':28s} {'fwd GFLOP/img':>13s} {'fwd MFU%':>9s} {'train(≈3x) MFU%':>16s}")
    for name, (m, x) in cases.items():
        f = count_fwd_flops(m, x) / 1e9
        fwd_tflops = f * a.samples_per_sec / 1e3   # GFLOP/img * img/s = GFLOP/s /1000 = TFLOPS
        print(f"{name:28s} {f:13.2f} {100*fwd_tflops/a.peak:9.2f} {100*3*fwd_tflops/a.peak:16.2f}")


if __name__ == "__main__":
    main()
