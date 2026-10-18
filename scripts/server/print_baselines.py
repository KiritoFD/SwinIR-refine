import json
from pathlib import Path

root = Path("/home/ds/realsr/experiments/diffusion/baselines_official")
subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

print(f"{'Model':15s} | {'Eval Type':12s} | {'PSNR-Y':8s} | {'SSIM-Y':8s} | {'PSNR-RGB':8s} | {'MUSIQ':7s} | {'MANIQA':7s}")
print("-" * 75)

for d in subdirs:
    for mode in ["eval_iqa", "eval_iqa_tta"]:
        ep = d / mode / "eval.json"
        if ep.exists():
            data = json.loads(ep.read_text(encoding="utf-8"))
            iqa = data.get("iqa_scores", {})
            musiq = iqa.get("musiq", 0.0)
            maniqa = iqa.get("maniqa", 0.0)
            print(f"{d.name:15s} | {mode:12s} | {data['psnr_y']:8.4f} | {data['ssim_y']:8.4f} | {data['psnr_rgb']:8.2f} | {musiq:7.3f} | {maniqa:7.4f}")
