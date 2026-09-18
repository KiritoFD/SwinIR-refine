"""Collect every official 100-pair eval into one table.

Sources differ in where they stored things (iqa_eval/, per-arm eval_official/,
per-arm eval_iqa/, the bicubic floor script), and not all of them have the IQA
scores, so gather them defensively and print one joined table.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/ds/realsr/experiments/diffusion")

# label -> candidate eval.json paths
SOURCES = [
    ("bicubic 地板", ["iqa_eval/bicubic_floor/eval.json"]),
    ("latent_flow_flux", ["iqa_eval/latent_flow_flux/eval.json", "latent_flow_flux/eval_official/eval.json"]),
    ("latent_flow_flux_XS", ["latent_flow_flux_XS/eval_official/eval.json"]),
    ("latent_reg_flux", ["iqa_eval/latent_reg_flux/eval.json", "latent_reg_flux/eval_official/eval.json"]),
    ("latent_flow_res", ["iqa_eval/latent_flow_res/eval.json", "latent_flow_res/eval_official/eval.json"]),
    ("pixel_flow (NFE16)", ["pixel_flow/eval_official/eval.json"]),
    ("DiT pixel_reg_XS", ["iqa_eval/dit_pixel_reg_XS/eval.json"]),
    ("DiT pixel_reg", ["iqa_eval/dit_pixel_reg/eval.json"]),
    ("s1_b32", ["iqa_eval/s1_b32/eval.json"]),
    ("s1_b40", ["iqa_eval/s1_b40/eval.json"]),
    ("s1_b48", ["iqa_eval/s1_b48/eval.json"]),
    ("s1_b64", ["iqa_eval/s1_b64/eval.json"]),
    ("s1_b64 +预训练", ["iqa_eval/s1_b64_pretrained/eval.json"]),
    ("s1_b64 +预训练+后训练", ["b64_ft15k/eval_iqa/eval.json"]),
    ("shape 1122_b96", ["shape_sweep/1122_b96/eval_iqa/eval.json"]),
    ("shape 1124_b80", ["shape_sweep/1124_b80/eval_iqa/eval.json"]),
    ("shape 1124_b96", ["shape_sweep/1124_b96/eval_iqa/eval.json"]),
    ("shape 1244_b80", ["shape_sweep/1244_b80/eval_iqa/eval.json"]),
    ("shape 1122_b80", ["shape_sweep/1122_b80/eval_iqa/eval.json"]),
]

print(f"{'实验':26s} {'n':>4s} {'Y':>8s} {'SSIM':>8s} {'RGB':>7s} {'MUSIQ':>8s} {'MANIQA':>8s}")
print("-" * 74)
for label, cands in SOURCES:
    data = None
    for c in cands:
        f = ROOT / c
        if f.is_file():
            try:
                data = json.loads(f.read_text())
                break
            except Exception:
                pass
    if data is None:
        print(f"{label:26s}     (缺)")
        continue
    iqa = data.get("iqa_scores", {})
    musiq = iqa.get("musiq")
    maniqa = iqa.get("maniqa")
    print(f"{label:26s} {data.get('n', 0):4d} {data.get('psnr_y', float('nan')):8.4f} "
          f"{data.get('ssim_y', float('nan')):8.4f} {data.get('psnr_rgb', float('nan')):7.2f} "
          f"{(f'{musiq:.3f}' if musiq else '    -'):>8s} {(f'{maniqa:.4f}' if maniqa else '      -'):>8s}",
          flush=True)
