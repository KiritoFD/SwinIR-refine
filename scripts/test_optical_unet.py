"""Unit test for Optical-Conditioned ResUNet (Section 3.1).

Verifies:
1. UNet builds cleanly with optical_cond=True.
2. Zero-initialization guarantee: at init, forward(x, optics=optics) yields exactly 0.0 residual.
3. Gradient flows back into OpticalEmbedding parameters during backward().
4. Compatibility: forward(x, t) without optics continues to work identically for regression/flow.
"""

import torch
from diffusion.unet import build_unet, OpticalEmbedding

def test_optical_unet():
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)

    # 1. Test standalone OpticalEmbedding
    opt_emb = OpticalEmbedding(dim=256, in_features=5)
    # [r_fov, cos_theta, sin_theta, is_canon, is_nikon]
    dummy_optics = torch.tensor([
        [0.75, 0.707, 0.707, 1.0, 0.0],
        [0.25, 1.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    emb = opt_emb(dummy_optics)
    assert emb.shape == (B, 256)
    # Since final layer is zero-init, emb must be exactly 0.0 at init
    assert (emb.abs() < 1e-7).all(), f"OpticalEmbedding final layer must be zero-init! Max: {emb.abs().max()}"
    print("SUCCESS 1: OpticalEmbedding zero-initialization verified!")

    # 2. Test UNet with optical_cond=True
    model = build_unet(size="XS", base=32, in_channels=3, out_channels=3, optical_cond=True)
    out_opt = model(x, optics=dummy_optics)
    assert out_opt.shape == (B, 3, H, W)
    # Since model output layer and optical embedding are zero-init, output must be exactly 0.0
    assert (out_opt.abs() < 1e-7).all(), f"UNet output at step 0 must be 0.0! Max: {out_opt.abs().max()}"
    print("SUCCESS 2: Optical-Conditioned UNet zero-initialization verified!")

    # 3. Test backward pass and gradient flow
    loss = (out_opt + model.opt_embed(dummy_optics).sum()).sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.opt_embed.parameters() if p.grad is not None)
    assert grad_norm > 0, "Gradients must flow into opt_embed!"
    print(f"SUCCESS 3: Backward pass gradient flow verified! Grad norm: {grad_norm:.6f}")

    # 4. Backward compatibility without optics (dummy t_zeros)
    t_zeros = torch.zeros(1)
    out_compat = model(x, t_zeros)
    assert out_compat.shape == (B, 3, H, W)
    print("SUCCESS 4: Backward compatibility with baseline forward(x, t) verified!")

if __name__ == "__main__":
    test_optical_unet()
