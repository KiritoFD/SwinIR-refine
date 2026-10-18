"""Automated benchmark leaderboard aggregator for WaveSR and official baselines.

Scans all evaluation results under experiments/ and produces a unified,
publication-quality comparison table across the full 8-metric suite.
"""

import json
import os
from pathlib import Path


def format_metric(val, fmt="{:.4f}"):
    if val is None or val == "" or val == "-":
        return "-"
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return str(val)


def collect_evaluations(root_dirs):
    records = []
    
    # 1. Preset benchmark targets
    presets = {
        "WaveSR-Base64 + WD-DCP + TTA (Ours)": "experiments/diffusion/mech_best/p_shift4/eval_wddcp_tta_full100/eval.json",
        "WaveSR-Base64 + WD-DCP (Ours)": "experiments/diffusion/mech_best/p_shift4/eval_wddcp_full100/eval.json",
        "WaveSR-Base64 (Raw)": "experiments/diffusion/mech_best/p_shift4/eval_raw_full100/eval.json",
        "Scale-96 Muon (Raw)": "experiments/diffusion/scale96_muon/eval_full_metrics_raw/eval.json",
        "Scale-96 Muon + WD-DCP": "experiments/diffusion/scale96_muon/eval_full_metrics_wddcp/eval.json",
        "Scale-96 Muon + WD-DCP + TTA": "experiments/diffusion/scale96_muon/eval_full_metrics_wddcp_tta/eval.json",
        "Scale-96 Muon + Adaptive CA-DCP + TTA": "experiments/diffusion/scale96_muon/eval_full_metrics_adaptive_ca_wddcp_tta/eval.json",
        "WaveSR-Physics (Raw)": "experiments/diffusion/wavesr_physics_b64/eval_raw/eval.json",
        "WaveSR-Physics + Adaptive CA-DCP": "experiments/diffusion/wavesr_physics_b64/eval_wddcp_adaptive/eval.json",
        "WaveSR-Physics + Adaptive CA-DCP + TTA": "experiments/diffusion/wavesr_physics_b64/eval_wddcp_adaptive_tta/eval.json",
        "SwinIR-L + TTA (Official)": "experiments/diffusion/baselines_official/swinir_off/eval_iqa_tta/eval.json",
        "SwinIR-L (Official)": "experiments/diffusion/baselines_official/swinir_off/eval_iqa/eval.json",
        "HAT-L + TTA (Official)": "experiments/diffusion/baselines_official/hat_off/eval_iqa_tta/eval.json",
        "HAT-L (Official)": "experiments/diffusion/baselines_official/hat_off/eval_iqa/eval.json",
        "RCAN + TTA (Official)": "experiments/diffusion/baselines_official/rcan_off/eval_iqa_tta/eval.json",
        "RCAN (Official)": "experiments/diffusion/baselines_official/rcan_off/eval_iqa/eval.json",
        "RDN + TTA (Official)": "experiments/diffusion/baselines_official/rdn_off/eval_iqa_tta/eval.json",
        "RDN (Official)": "experiments/diffusion/baselines_official/rdn_off/eval_iqa/eval.json",
        "RLFN + TTA (8-fold)": "experiments/diffusion/baselines/rlfn_x2/eval_iqa_tta/eval.json",
        "RLFN (Raw)": "experiments/diffusion/baselines/rlfn_x2/eval_iqa/eval.json",
        "SRFormer (Official)": "experiments/diffusion/baselines_official/srformer_off/eval_iqa/eval.json",
        "EDSR (Official)": "experiments/diffusion/baselines_official/edsr_off/eval_iqa/eval.json",
        "VDSR (Official)": "experiments/diffusion/baselines_official/vdsr_off/eval_iqa/eval.json",
    }
    
    for name, p_str in presets.items():
        p = Path(p_str)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                iqa = d.get("iqa_scores", {})
                records.append({
                    "method": name,
                    "psnr_y": d.get("psnr_y"),
                    "ssim_y": d.get("ssim_y"),
                    "psnr_rgb": d.get("psnr_rgb"),
                    "lpips": iqa.get("lpips"),
                    "dists": iqa.get("dists"),
                    "niqe": iqa.get("niqe"),
                    "musiq": iqa.get("musiq"),
                    "maniqa": iqa.get("maniqa"),
                    "n_pairs": d.get("n", 100),
                    "path": str(p),
                })
            except Exception as e:
                pass
    return records


def print_leaderboard():
    records = collect_evaluations(["experiments"])
    if not records:
        print("No evaluation records found.")
        return
        
    # Sort by PSNR-Y descending
    records.sort(key=lambda x: x["psnr_y"] if x["psnr_y"] is not None else -1.0, reverse=True)
    
    header = (
        f"{'Method':40s} | {'PSNR-Y':8s} | {'SSIM-Y':8s} | {'RGB':6s} | "
        f"{'LPIPS':7s} | {'DISTS':7s} | {'NIQE':7s} | {'MUSIQ':7s} | {'MANIQA':7s}"
    )
    sep = "-" * len(header)
    
    print("\n" + "=" * len(header))
    print(" WaveSR Official Benchmark Leaderboard (A-Strict, RealSR x2 Full 100 Pairs)")
    print("=" * len(header))
    print(header)
    print(sep)
    
    md_lines = [
        "# WaveSR Official Benchmark Leaderboard (A-Strict, RealSR x2)\n",
        "| Method | PSNR-Y (dB) | SSIM-Y | PSNR-RGB (dB) | LPIPS ↓ | DISTS ↓ | NIQE ↓ | MUSIQ ↑ | MANIQA ↑ |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    
    for r in records:
        py = format_metric(r['psnr_y'], "{:8.4f}")
        sy = format_metric(r['ssim_y'], "{:8.4f}")
        prgb = format_metric(r['psnr_rgb'], "{:6.2f}")
        lpips = format_metric(r['lpips'], "{:7.4f}")
        dists = format_metric(r['dists'], "{:7.4f}")
        niqe = format_metric(r['niqe'], "{:7.3f}")
        musiq = format_metric(r['musiq'], "{:7.3f}")
        maniqa = format_metric(r['maniqa'], "{:7.4f}")
        
        print(f"{r['method']:40s} | {py} | {sy} | {prgb} | {lpips} | {dists} | {niqe} | {musiq} | {maniqa}")
        md_lines.append(f"| **{r['method']}** | **{py.strip()}** | {sy.strip()} | {prgb.strip()} | {lpips.strip()} | {dists.strip()} | {niqe.strip()} | {musiq.strip()} | {maniqa.strip()} |")
        
    print(sep + "\n")
    
    # Save to experiments/BENCHMARK_LEADERBOARD.md
    out_md = Path("experiments/BENCHMARK_LEADERBOARD.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved Markdown leaderboard to {out_md}")


if __name__ == "__main__":
    print_leaderboard()
