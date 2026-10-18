#!/usr/bin/env bash
# Re-run only the tiling-correctness gate (whole-image vs tiled forward must agree).
cd /home/ds/realsr || exit 1
P=/home/ds/miniconda3/envs/harness-qwen/bin/python
find diffusion scripts/server -name '*.py' -o -name '*.sh' 2>/dev/null | xargs -r sed -i 's/\r$//'
$P - <<'PY'
import torch, torch.nn.functional as F, numpy as np
from diffusion.data import build_index
from diffusion.eval_official import sr_lr_tiled, sr_pixel_tiled, to_tensor, load_rgb_u8
from diffusion import baselines_official as bo

pairs = build_index("data/RealSR(V3)", ("Canon", "Nikon"), "Test", 2)
lr = load_rgb_u8(pairs[0][0])[:448, :448]
print("LR crop", lr.shape, "-> HR", lr.shape[0]*2)
for name in bo.OFFICIAL:
    try:
        m = bo.build(name, scale=2).to("cuda").eval()
        space = getattr(m, "net_input", "lr")
        x = to_tensor(lr).unsqueeze(0).cuda()
        if space == "hr":
            x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.no_grad():
            full = m(x, torch.zeros(1, device="cuda")).clamp(0, 1)
        if space == "lr":
            t = sr_lr_tiled(m, lr, 2, 64, 16, "cuda", tile_batch=2)
        else:
            t = sr_pixel_tiled(m, lr, 2, 8, 64, 16, False, "cuda", "reg", 1234, tile_batch=2)
        tiled = to_tensor(t).unsqueeze(0).cuda()
        h = min(full.shape[-2], tiled.shape[-2]); w = min(full.shape[-1], tiled.shape[-1])
        mse = F.mse_loss(full[:, :, :h, :w], tiled[:, :, :h, :w]).item()
        psnr = 99.0 if mse <= 1e-12 else -10 * np.log10(mse)
        print("  %-12s [%s-in] full%s tiled%s win=%s -> %.2f dB  %s"
              % (name, space, tuple(full.shape), tuple(tiled.shape), m.win, psnr,
                 "OK" if psnr > 30 else "*** TILING BUG ***"))
        del m, full, tiled; torch.cuda.empty_cache()
    except Exception as e:
        print("  %-12s FAILED %s: %s" % (name, type(e).__name__, str(e)[:110]))
PY
