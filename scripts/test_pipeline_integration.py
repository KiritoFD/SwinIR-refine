import torch
import torch.nn.functional as F
from diffusion.unet import build_unet
from diffusion.physics_loss import ll_swap_target, bounded_align_target, chromatic_aberration_loss

def test_pipeline_integration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline integration test on {device}...")
    
    # 1. Instantiate UNet with optical conditioning
    model = build_unet(
        "S",
        base=32,
        mult=(1, 2, 2),
        num_res=1,
        attn_levels=(),
        in_stride=1,
        out_scale=1,
        align=2,
        residual=True,
        optical_cond=True
    ).to(device)
    
    B, C, H, W = 4, 3, 64, 64
    lr_up = torch.rand(B, C, H, W, device=device)
    hr = torch.rand(B, C, H, W, device=device)
    
    # 5D optics tensor: [r_fov, cos_theta, sin_theta, is_canon, is_nikon]
    optics = torch.tensor([
        [0.2, 0.98, 0.19, 1.0, 0.0],
        [0.8, 0.70, 0.71, 1.0, 0.0],
        [0.5, 0.50, 0.86, 0.0, 1.0],
        [0.9, -0.6, 0.80, 0.0, 1.0],
    ], device=device, dtype=torch.float32)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    t_zeros = torch.zeros(1, device=device)
    
    from diffusion.physics_loss import ycbcr_weighted_l1_loss, chroma_tv_loss
    
    # Run 4 warmup steps to verify gradient propagation through zero-initialized output and residual layers
    for s in range(4):
        res = model(lr_up, t_zeros.expand(B), optics=optics)
        pred = (lr_up + res).clamp(0, 1)
        target_ll = ll_swap_target(lr_up, hr)
        target_align = bounded_align_target(pred, target_ll, max_shift=0.5)
        
        # Combined physical loss suite
        loss_main = ycbcr_weighted_l1_loss(pred, target_align, y_weight=1.0, chroma_weight=0.2)
        loss_ca = chromatic_aberration_loss(pred)
        loss_tv = chroma_tv_loss(pred)
        loss = loss_main + 0.01 * loss_ca + 0.005 * loss_tv
        
        opt.zero_grad()
        loss.backward()
        opt_grad_norm = sum(p.grad.norm().item() for p in model.opt_embed.parameters() if p.grad is not None)
        print(f"  [OK] Step {s}: loss={loss.item():.4f} (main={loss_main.item():.4f}, ca={loss_ca.item():.4f}, tv={loss_tv.item():.4f}), OpticalEmbedding grad norm={opt_grad_norm:.6f}")
        opt.step()
    
    assert opt_grad_norm > 0, "Optical embedding must receive non-zero gradients once output layer warms up!"
    print("All pipeline integration tests passed perfectly!")

if __name__ == "__main__":
    test_pipeline_integration()
