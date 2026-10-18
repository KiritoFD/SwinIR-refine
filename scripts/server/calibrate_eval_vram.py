import torch
from diffusion.eval_official import sr_pixel_tiled
from diffusion.unet import build_unet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
print(f"Total GPU Memory: {total_mem:.2f} GB")

# Create a realistic RealSR image: 1000 x 1500 LR (x2 -> 2000 x 3000 HR)
lr_u8 = np.random.randint(0, 256, (1000, 1500, 3), dtype=np.uint8)

model = build_unet("S", input_size=128, in_channels=3, base=64, mult=(1, 2, 4, 4),
                   num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1, align=64).to(device)
model.eval()

for tb in [16, 32, 64, 96, 128]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sr_u8 = sr_pixel_tiled(
            model, lr_u8, scale=2, steps=20, tile=64, pad=16, residual=True,
            device=device, objective="reg", seed=1234, tile_batch=tb, bf16=True
        )
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"EVAL: tile_batch={tb:3d} -> Peak VRAM: {peak_mem:5.2f} GB | Output SR shape: {sr_u8.shape}")
        del sr_u8
    except torch.cuda.OutOfMemoryError:
        print(f"EVAL: tile_batch={tb:3d} -> OOM!")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"EVAL Error: {e}")
        break
