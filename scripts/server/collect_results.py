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
    ("E1 SwinIR-light 0.61M", ["iqa_eval/E1_swinir_light/eval.json"]),
    ("E2 SwinIR-largeish 3.96M", ["iqa_eval/E2_swinir_capmatch/eval.json"]),
    ("A0 Mod pure L1 4.05M", ["iqa_eval/A0_l1_only/eval.json"]),
    ("E11 Mod+Align 4.05M", ["iqa_eval/E11_align_loss/eval.json"]),
    ("latent_flow_flux", ["iqa_eval/latent_flow_flux/eval.json", "latent_flow_flux/eval_official/eval.json"]),
    ("latent_flow_res", ["iqa_eval/latent_flow_res/eval.json", "latent_flow_res/eval_official/eval.json"]),
    ("latent_reg_flux", ["iqa_eval/latent_reg_flux/eval.json", "latent_reg_flux/eval_official/eval.json"]),
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
    ("cap b128", ["capacity/b128/eval_iqa/eval.json"]),
    ("cap b128+FFN", ["capacity/b128_ffn/eval_iqa/eval.json"]),
]

print(f"{'实验':26s} {'n':>4s} {'Y':>8s} {'SSIM':>8s} {'RGB':>7s} {'MUSIQ':>8s} {'MANIQA':>8s}")
print("-" * 74)
for label, cands in SOURCES:
    # Some IQA re-runs lost a few pairs (OOM retry), which moves the mean by more
    # than 1 dB on the latent arms.  Take Y/SSIM from the eval with the most pairs
    # and the IQA scores from whichever file actually has them.
    blobs = []
    for c in cands:
        f = ROOT / c
        if f.is_file():
            try:
                blobs.append(json.loads(f.read_text()))
            except Exception:
                pass
    if not blobs:
        print(f"{label:26s}     (缺)")
        continue
    data = max(blobs, key=lambda d: d.get("n", 0))
    iqa = {}
    for d in blobs:
        if d.get("iqa_scores"):
            iqa = dict(d["iqa_scores"])
            break
    musiq = iqa.get("musiq")
    maniqa = iqa.get("maniqa")
    print(f"{label:26s} {data.get('n', 0):4d} {data.get('psnr_y', float('nan')):8.4f} "
          f"{data.get('ssim_y', float('nan')):8.4f} {data.get('psnr_rgb', float('nan')):7.2f} "
          f"{(f'{musiq:.3f}' if musiq else '    -'):>8s} {(f'{maniqa:.4f}' if maniqa else '      -'):>8s}",
          flush=True)
