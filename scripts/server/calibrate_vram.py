import torch
import torch.nn as nn
from diffusion.unet import build_unet

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
if device == "cuda":
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU Memory: {total_mem:.2f} GB")

# Test 1: Checkpointing with larger batch / larger base
print("\n=== Testing with use_checkpoint=True ===")
for base in [64, 96]:
    for bs in [128, 160, 192, 224, 256]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_unet(
                "S", input_size=128, in_channels=3, base=base, mult=(1, 2, 4, 4),
                num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1, align=64,
                use_checkpoint=True
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(bs, 3, 128, 128, device=device)
            target = torch.randn(bs, 3, 128, 128, device=device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, torch.zeros(bs, device=device))
                loss = (out - target).abs().mean()
            loss.backward()
            opt.step()
            
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"CHECKPOINT: base={base:2d} | batch={bs:3d} | patch=64 -> Peak VRAM: {peak_mem:5.2f} GB")
            del model, opt, x, target, out, loss
        except torch.cuda.OutOfMemoryError:
            print(f"CHECKPOINT: base={base:2d} | batch={bs:3d} -> OOM!")
            del model, opt, x, target
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"Error: {e}")
            break

# Test 2: In-between batch sizes without checkpointing
print("\n=== Testing bs between 128 and 144 without checkpointing ===")
for bs in [130, 132, 136, 140]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build_unet(
            "S", input_size=128, in_channels=3, base=64, mult=(1, 2, 4, 4),
            num_res=2, attn_levels=(2, 3), in_stride=1, out_scale=1, align=64,
            use_checkpoint=False
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(bs, 3, 128, 128, device=device)
        target = torch.randn(bs, 3, 128, 128, device=device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, torch.zeros(bs, device=device))
            loss = (out - target).abs().mean()
        loss.backward()
        opt.step()
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"NO-CKPT: base=64 | batch={bs:3d} -> Peak VRAM: {peak_mem:5.2f} GB")
        del model, opt, x, target, out, loss
    except torch.cuda.OutOfMemoryError:
        print(f"NO-CKPT: base=64 | batch={bs:3d} -> OOM!")
        del model, opt, x, target
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"Error: {e}")
        break

