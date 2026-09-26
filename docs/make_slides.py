"""Generate docs/slides.pptx (16:9) summarising the RealSR project report.
Run:  python docs/make_slides.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

OUT = Path(__file__).resolve().parent / "slides.pptx"
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x22, 0x22, 0x22)


def slide(title, bullets, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, ACCENT
    if subtitle:
        p2 = tb.text_frame.add_paragraph()
        p2.text = subtitle
        p2.font.size, p2.font.color.rgb = Pt(14), RGBColor(0x60, 0x60, 0x60)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(12.1), Inches(5.4))
    tf = body.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = b
        para.font.size = Pt(18)
        para.font.color.rgb = DARK
        para.space_after = Pt(8)
    return s


def table_slide(title, header, rows, subtitle=None, col_w=None):
    s = slide(title, [], subtitle)
    shp = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.5), Inches(1.8),
                             Inches(12.3), Inches(0.5) * (len(rows) + 1))
    t = shp.table
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(14)
        c.text_frame.paragraphs[0].font.bold = True
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            c.text_frame.paragraphs[0].font.size = Pt(13)
    return s


# 1  title
s = prs.slides.add_slide(BLANK)
box = s.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(3.2))
tf = box.text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "Orthogonal-Frequency-Regularized Discriminative\nRegression for Real-World SR"
tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(36), True, ACCENT
p = tf.add_paragraph()
p.text = "RealSR (V3) · ×2 · official Test.m protocol"
p.font.size, p.font.color.rgb = Pt(20), RGBColor(0x60, 0x60, 0x60)
p = tf.add_paragraph()
p.text = "PSNR-Y 34.41 · SSIM 0.9285 · MUSIQ 55.96 · MANIQA 0.3534"
p.font.size, p.font.bold, p.font.color.rgb = Pt(26), True, DARK

# 2  problem
slide("Problem & the honest protocol",
      ["RealSR = real camera zoom pairs (Canon 5D3 / Nikon D810); 390 train, 100 test.",
       "Numbers are NOT comparable across papers — protocol differs:",
       "   • Limited-range Y vs full-range RGB (Y is 0.4–0.8 dB lower)",
       "   • border shave, test-subset choice, Protocol A (trained on RealSR) vs B (pretrained blind)",
       "We report the STRICTEST form: Limited-range Y, uint8, modcrop4, NO shave, all 100 pairs.",
       "Every external number in this deck carries a protocol tag."])

# 3  method
slide("Method",
      ["Stride-1 U-Net regressor (no lossy downsample) + residual-bicubic, zero-init head → step-0 = bicubic.",
       "Wavelet high-frequency loss: multi-scale L1 on orthogonal Haar HL/LH/HH subbands (λ≈8).",
       "Shift-ensemble variant (avg over dyadic offsets) ≈ shift-invariance — best perceptual arm.",
       "Muon optimizer (Newton–Schulz orthogonalised updates; AdamW on 1-D), lr 5e-3.",
       "BSRGAN synthetic-degradation pretraining → RealSR finetune; D4 8-fold TTA at inference."])

# 4  results
table_slide("Results — official paired protocol (100 pairs)",
            ["Method", "Type", "Params", "PSNR-Y", "SSIM", "MUSIQ", "MANIQA"],
            [["Bicubic", "floor", "—", "31.73", "0.8876", "39.40", "0.278"],
             ["SwinIR-light (re-trained)", "Transformer", "0.61M", "32.88", "0.9026", "45.43", "0.302"],
             ["SwinIR-largeish (re-trained)", "Transformer", "3.96M", "32.97", "0.9058", "46.79", "0.305"],
             ["E11 Mod-SwinIR (Align-L1)", "Transformer", "4.05M", "33.47", "0.9143", "49.15", "0.309"],
             ["EDSR-baseline (repro.)", "CNN", "1.19M", "33.60", "0.9167", "53.14", "0.327"],
             ["latent rectified-flow (ours)", "Generative", "32.9M", "28.36", "0.790", "50.12", "0.224"],
             ["Ours (full recipe + TTA)", "U-Net stride-1", "18.7M", "34.41", "0.9285", "55.96", "0.3534"]],
            subtitle="vs re-trained SwinIR: +1.44 dB · +0.023 SSIM · +9.2 MUSIQ · +0.048 MANIQA")

# 5  baselines
slide("Baselines: reproduced vs cited",
      ["Reproduced in-repo on the SAME protocol (A-strict): SwinIR (light/largeish), EDSR-baseline;",
       "   RCAN / SRResNet / RRDB in the reproduction pipeline (docs/baseline).",
       "Cited with protocol tags (not compared): BSRGAN/Real-ESRGAN (D-blind);",
       "   StableSR/DiffBIR/SeeSR/ResShift/SinSR/OSEDiff/AdcSR/TSD-SR/TinySR/PURE/RealSR-R1 (B/C protocols).",
       "Their RealSR PSNR sits at 22.6–26.3 dB on a synthetically re-degraded input — a different task."])

# 6  ablations
table_slide("Ablations (Δ PSNR-Y)",
            ["Change", "Δ PSNR-Y", "Perceptual", "Cost"],
            [["stride-2 → stride-1 stem", "+0.38", "↑", "fewer params"],
             ["+ wavelet-HF loss (λ8)", "+0.11", "MUSIQ/MANIQA ↑", "zero inference"],
             ["+ shift-ensemble", "+0.02", "MUSIQ 56.0 (best)", "zero inference"],
             ["+ Muon (5e-3)", "+0.15", "↑", "zero inference"],
             ["+ BSRGAN pretrain", "+0.09", "↑", "6× wall-clock"],
             ["+ TTA", "+0.10", "SSIM/MANIQA ↑", "8× inference"],
             ["capacity 5M → 72.5M", "saturates", "~", "linear"]])

# 7  ceiling
table_slide("The empirical ceiling (training-free, no model)",
            ["wall", "×2", "×3", "×4"],
            [["Registration E_align (0.3px shift)", "37.7", "37.7", "37.7"],
             ["Band-limit E_null (LP to LR Nyquist)", "40.65", "34.55", "31.27"],
             ["Sensor noise E_noise (MAD σ≈0.9/255)", "49.05", "49.05", "49.05"],
             ["composite (bandlimit+noise)", "40.1", "34.4", "31.2"],
             ["+ 0.3px residual registration", "35.7", "~32", "~29.5"]],
            subtitle="×2 is registration-limited; our 34.41 sits inside 34.4–35.7. Corrects the '38–40 dB noise floor' folklore.")

# 8  tradeoff
slide("Perception–distortion tradeoff, reproduced in-house",
      ["Same test set, same protocol: regression 34.41 dB / 0.9285 SSIM  vs  generative 28.36 dB / 0.790 SSIM.",
       "Generative trades ~6 dB fidelity for higher no-reference IQA (Blau & Michaeli, CVPR 2018).",
       "On RealSR: the fidelity frontier is capped by physics (registration),",
       "the realism frontier is capped by the 390-pair data budget.",
       "→ Practical conclusion: PSNR has no sustainable headroom at ×2; the future is perceptual."])

# 9  negatives
slide("Negative results (all evaluated, all reported)",
      ["Reporting 28+ runs that did NOT help — same protocol, honest verdicts.",
       "Mamba/VSS: −2.3 dB at equal sample budget (memory-bound scan, poor sample efficiency).",
       "Dual-branch wavelet U-Net, DTCWT basis, D4 equivariance, coord injection, freq-routing, gated FFN.",
       "Frequency-domain rectified flow (HF residual generation): monotone image destruction; judged negative.",
       "Adversarial degradation mining (N3, with Muon): neutral — does not break the plateau."])

# 10 conclusion
slide("Conclusions",
      ["A recipe wins: stride-1 + wavelet-HF(+shift) + Muon + BSRGAN pretrain + TTA → 34.41 dB / 0.9285 SSIM.",
       "PSNR at ×2 is physics-capped (~34.4–35.7 dB): registration, not noise, is the wall.",
       "×3/×4 are lighter-tuned and still have headroom — the open next step on the fidelity axis.",
       "Beyond that: perceptual metrics (MUSIQ/LPIPS/FID) are where the remaining gains live.",
       "Full chronicle: exp/FINAL_REPORT.md · baselines: docs/baseline/ · ceiling tool: diffusion/oracle.py"])

# 11 references
slide("References",
      ["Cai et al., RealSR / LP-KPN, ICCV 2019.",
       "Liang et al., SwinIR, ICCV-W 2021.  Lim et al., EDSR, CVPR-W 2017.  Zhang et al., RCAN, ECCV 2018.",
       "Wang et al., ESRGAN, ECCV-W 2018.  Zhang et al., BSRGAN, ICCV 2021.  Wang et al., Real-ESRGAN, ICCV-W 2021.",
       "Qiao et al., RealSR-R1, arXiv:2506.16796, 2025.  TinySR, arXiv:2508.17434, 2025.",
       "Blau & Michaeli, Perception-Distortion Tradeoff, CVPR 2018 (DOI 10.1109/CVPR.2018.00652).",
       "Baker & Kanade, Limits on SR, TPAMI 2002 (DOI 10.1109/TPAMI.2002.1033210).",
       "Robinson & Milanfar, TIP 2004/2006.  Chatterjee & Milanfar, TIP 2010."])

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
