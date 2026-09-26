"""Generate paper-grade figures for the RealSR report/deck.
Run:  python docs/make_figs.py   ->  docs/figs/*.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
    "grid.alpha": 0.28, "axes.edgecolor": "#555", "figure.dpi": 150,
})
ACC, BLUE, GRAY, GEN, GREEN = "#1f4e79", "#3a6ea5", "#8a8a8a", "#c0392b", "#2f7d4f"

# ---- data (A-strict, official protocol) ------------------------------------
names = ["Bicubic", "SwinIR\nlight", "SwinIR\nlargeish", "E11\nAlign-L1", "EDSR\nbaseline", "s1_b64\nanchor", "Ours"]
psnr = [31.73, 32.88, 32.97, 33.47, 33.60, 34.11, 34.41]
ssim = [0.8876, 0.9026, 0.9058, 0.9143, 0.9167, 0.9246, 0.9285]
colors = [GRAY] * 5 + [BLUE, ACC]

# fig 1: results bars
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax[0].bar(names, psnr, color=colors)
ax[0].axhline(34.41, ls="--", c=GEN, lw=1.2)
ax[0].set_ylabel("PSNR-Y (dB)")
ax[0].set_title("Fidelity — official paired protocol")
ax[0].set_ylim(31.0, 35.0)
for i, v in enumerate(psnr):
    ax[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
ax[1].bar(names, ssim, color=colors)
ax[1].set_ylabel("SSIM")
ax[1].set_title("Structure — official paired protocol")
ax[1].set_ylim(0.88, 0.94)
for i, v in enumerate(ssim):
    ax[1].text(i, v + 0.0012, f"{v:.4f}", ha="center", fontsize=8)
for a in ax:
    a.tick_params(axis="x", labelsize=8.5)
fig.suptitle("Real-World SR on RealSR ×2 (Limited-range Y, no shave, 100 pairs)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig_results.png", bbox_inches="tight")
plt.close(fig)

# fig 2: ceiling
shifts = np.array([0.2, 0.3, 0.5, 1.0])
align_y = np.array([41.11, 37.69, 33.41, 27.94])
scales = ["×2", "×3", "×4"]
enull = [40.65, 34.55, 31.27]
comp = [40.1, 34.4, 31.2]
ours_psnr = [34.41, 31.13, 29.51]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax[0].plot(shifts, align_y, "o-", c=ACC, lw=2)
ax[0].fill_between(shifts, align_y, 20, color=ACC, alpha=0.08)
ax[0].axhline(34.41, ls="--", c=GEN, lw=1.4)
ax[0].text(0.98, 34.7, "ours 34.41 dB", color=GEN, fontsize=9, ha="right")
ax[0].set_xlabel("residual registration error (px)")
ax[0].set_ylabel("PSNR-Y (dB)")
ax[0].set_title("Wall 1 — registration penalty")
ax[0].set_ylim(25, 43)
x = np.arange(3)
w = 0.38
ax[1].bar(x - w / 2, enull, w, label="band-limit $E_{null}$", color=ACC)
ax[1].bar(x + w / 2, comp, w, label="band-limit + noise", color=BLUE)
ax[1].scatter(x, ours_psnr, marker="D", s=64, c=GEN, label="our model", zorder=5)
for i in range(3):
    ax[1].text(i - w / 2, enull[i] + 0.3, f"{enull[i]:.1f}", ha="center", fontsize=8.5)
    ax[1].text(i + w / 2, comp[i] + 0.3, f"{comp[i]:.1f}", ha="center", fontsize=8.5)
    ax[1].text(i, ours_psnr[i] - 1.6, f"{ours_psnr[i]:.2f}", ha="center", fontsize=8.5, color=GEN)
ax[1].set_xticks(x)
ax[1].set_xticklabels(scales)
ax[1].set_ylabel("PSNR-Y (dB)")
ax[1].set_title("Walls 2+3 — band-limit & noise ceiling")
ax[1].set_ylim(24, 50)
ax[1].legend(fontsize=8.5, loc="upper right")
fig.suptitle("Training-free decomposition of the RealSR PSNR ceiling", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig_ceiling.png", bbox_inches="tight")
plt.close(fig)

# fig 3: ablation
ab_names = ["stride-2→\nstride-1", "wavelet-HF\nloss λ8", "shift\nensemble", "Muon\n5e-3", "BSRGAN\npretrain", "TTA\n(x8)", "capacity\n(5→72M)"]
ab_delta = [0.38, 0.11, 0.02, 0.15, 0.09, 0.10, 0.05]
fig, ax = plt.subplots(figsize=(9.6, 4.4))
b = ax.bar(ab_names, ab_delta, color=[ACC, ACC, GREEN, ACC, GRAY, GRAY, GRAY])
ax.set_ylabel("Δ PSNR-Y (dB)")
ax.set_title("What actually moves RealSR ×2 (one change at a time, same protocol)", fontweight="bold")
for r, v in zip(b, ab_delta):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.006, f"+{v:.2f}", ha="center", fontsize=9)
ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
plt.close(fig)

# fig 4: perception-distortion scatter
pts = {
    "Bicubic": (31.73, 39.40, GRAY),
    "SwinIR-largeish": (32.97, 46.79, GRAY),
    "E11": (33.47, 49.15, GRAY),
    "EDSR": (33.60, 53.14, GRAY),
    "s1_b64 anchor": (34.11, 55.22, BLUE),
    "Ours (regression)": (34.41, 55.96, ACC),
    "latent flow (generative)": (28.36, 50.12, GEN),
}
fig, ax = plt.subplots(figsize=(9.6, 5.4))
for name, (x, y, c) in pts.items():
    ax.scatter(x, y, s=95, c=c, zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
ax.annotate("", xy=(28.7, 49.0), xytext=(34.0, 55.6), arrowprops=dict(arrowstyle="->", color="#555", ls="--", lw=1.4))
ax.text(31.3, 48.4, "generative:\nfidelity ↓ / realism ↑", fontsize=9, color=GEN)
ax.set_xlabel("PSNR-Y (dB)   →  fidelity")
ax.set_ylabel("MUSIQ   →  no-reference perception")
ax.set_title("Perception–distortion tradeoff on RealSR ×2 (Blau & Michaeli, CVPR 2018)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_tradeoff.png", bbox_inches="tight")
plt.close(fig)

print("wrote figs:", *[p.name for p in sorted(OUT.glob("*.png"))])
