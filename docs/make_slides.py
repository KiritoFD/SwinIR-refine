"""Generate docs/slides.pptx (16:9) for the RealSR project report — styled, with
embedded figures (run docs/make_figs.py first to create docs/figs/*.png).
Run:  python docs/make_slides.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
OUT = HERE / "slides.pptx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_LT = RGBColor(0x3A, 0x6E, 0xA5)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
DARK = RGBColor(0x22, 0x22, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OURROW = RGBColor(0xFB, 0xE5, 0xA7)  # highlight for "Ours"
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
_n = {"i": 0}


def _chrome(s, kicker=None):
    """footer + page number on every content slide."""
    _n["i"] += 1
    fb = s.shapes.add_textbox(Inches(0.5), Inches(7.02), Inches(9.5), Inches(0.35))
    p = fb.text_frame.paragraphs[0]
    p.text = "RealSR ×2 · Real-World SR · stride-1 + wavelet-HF + Muon + TTA"
    p.font.size, p.font.color.rgb = Pt(9), RGBColor(0x88, 0x88, 0x88)
    pb = s.shapes.add_textbox(Inches(12.3), Inches(7.02), Inches(0.9), Inches(0.35))
    pp = pb.text_frame.paragraphs[0]
    pp.text = str(_n["i"])
    pp.alignment = PP_ALIGN.RIGHT
    pp.font.size, pp.font.color.rgb = Pt(10), RGBColor(0x88, 0x88, 0x88)


def title_bar(s, title, subtitle=None):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(1.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    tf = bar.text_frame; tf.word_wrap = True
    tf.margin_left, tf.margin_top = Inches(0.45), Inches(0.12)
    p = tf.paragraphs[0]; p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(26), True, WHITE
    if subtitle:
        q = tf.add_paragraph(); q.text = subtitle
        q.font.size, q.font.color.rgb = Pt(13), RGBColor(0xC7, 0xD6, 0xE5)
    return bar


def bullets_slide(title, bullets, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, subtitle)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.45), Inches(12.1), Inches(5.4))
    tf = body.text_frame; tf.word_wrap = True
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        lvl = 0
        if b.startswith("  "):
            b = b.strip(); lvl = 1
        para.text = b
        para.level = lvl
        para.font.size = Pt(17 if lvl == 0 else 15)
        para.font.color.rgb = DARK if lvl == 0 else RGBColor(0x44, 0x44, 0x44)
        para.space_after = Pt(9 if lvl == 0 else 4)
    _chrome(s)
    return s


def table_slide(title, header, rows, subtitle=None, highlight=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, subtitle)
    nr, nc = len(rows) + 1, len(header)
    top = Inches(1.5)
    height = Inches(0.42) * nr
    shp = s.shapes.add_table(nr, nc, Inches(0.4), top, Inches(12.5), height)
    t = shp.table
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = ACCENT
        c.margin_top = c.margin_bottom = Emu(45720)
        pr = c.text_frame.paragraphs[0]; pr.text = h
        pr.font.size, pr.font.bold, pr.font.color.rgb = Pt(12), True, WHITE
    for i, r in enumerate(rows, start=1):
        is_ours = highlight is not None and highlight(r)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = OURROW if is_ours else (LIGHT if i % 2 == 0 else WHITE)
            pr = c.text_frame.paragraphs[0]; pr.text = str(v)
            pr.font.size = Pt(12)
            pr.font.bold = is_ours
            pr.font.color.rgb = DARK
    _chrome(s)
    return s


def image_slide(title, img, caption=None, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, subtitle)
    p = str(FIGS / img)
    if Path(p).exists():
        pic = s.shapes.add_picture(p, Inches(0.7), Inches(1.45), width=Inches(11.9))
        # center vertically-ish if short
        top = pic.top
        if caption:
            cb = s.shapes.add_textbox(Inches(0.7), Inches(6.55), Inches(11.9), Inches(0.4))
            cp = cb.text_frame.paragraphs[0]; cp.text = caption; cp.alignment = PP_ALIGN.CENTER
            cp.font.size, cp.font.color.rgb = Pt(11), RGBColor(0x60, 0x60, 0x60)
    else:
        bullets_slide(title, [f"(missing figure {img} — run docs/make_figs.py)"], subtitle)
        return s
    _chrome(s)
    return s


# ---- 1 title ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.0), SW, Inches(3.5))
band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()
tf = band.text_frame; tf.word_wrap = True
tf.margin_left = Inches(0.8); tf.margin_top = Inches(0.35)
tf.paragraphs[0].text = "Orthogonal-Frequency-Regularized Discriminative Regression"
tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(30), True, WHITE
p = tf.add_paragraph(); p.text = "for Real-World Single-Image Super-Resolution"
p.font.size, p.font.bold, p.font.color.rgb = Pt(26), True, WHITE
p = tf.add_paragraph(); p.text = " "
p = tf.add_paragraph(); p.text = "RealSR (V3) · ×2 · official Test.m (Limited-range Y, no shave, 100 pairs)"
p.font.size, p.font.color.rgb = Pt(14), RGBColor(0xC7, 0xD6, 0xE5)
tb = s.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.3), Inches(0.9))
tbp = tb.text_frame.paragraphs[0]
tbp.text = "PSNR-Y 34.41   ·   SSIM 0.9285   ·   MUSIQ 55.96   ·   MANIQA 0.3534   (18.7M params)"
tbp.alignment = PP_ALIGN.CENTER
tbp.font.size, tbp.font.bold, tbp.font.color.rgb = Pt(20), True, ACCENT

# ---- 2 results figure ------------------------------------------------------
image_slide("Results on the official paired protocol", "fig_results.png",
            subtitle="Ours tops every fidelity/structure metric; dashed line = our result",
            caption="Reproduced baselines share our exact A-strict protocol.  vs re-trained SwinIR: +1.44 dB / +0.023 SSIM / +9.2 MUSIQ / +0.048 MANIQA")

# ---- 3 results table -------------------------------------------------------
table_slide("Results — official paired protocol (100 pairs)",
            ["Method", "Type", "Params", "PSNR-Y", "SSIM", "MUSIQ", "MANIQA"],
            [["Bicubic", "floor", "—", "31.73", "0.8876", "39.40", "0.278"],
             ["SwinIR-light (re-trained)", "Transformer", "0.61M", "32.88", "0.9026", "45.43", "0.302"],
             ["SwinIR-largeish (re-trained)", "Transformer", "3.96M", "32.97", "0.9058", "46.79", "0.305"],
             ["E11 Mod-SwinIR (Align-L1)", "Transformer", "4.05M", "33.47", "0.9143", "49.15", "0.309"],
             ["EDSR-baseline (repro.)", "CNN", "1.19M", "33.60", "0.9167", "53.14", "0.327"],
             ["s1_b64 (AdamW, zero-pretrain)", "U-Net", "18.7M", "34.11", "0.9246", "55.22", "0.342"],
             ["latent rectified-flow (ours)", "Generative", "32.9M", "28.36", "0.790", "50.12", "0.224"],
             ["Ours (full recipe + TTA)", "U-Net stride-1", "18.7M", "34.41", "0.9285", "55.96", "0.3534"]],
            highlight=lambda r: r[0].startswith("Ours"),
            subtitle="Limited-range Y · uint8 · modcrop4 · no shave")

# ---- 4 method --------------------------------------------------------------
bullets_slide("Method — one recipe, four levers",
              ["Stride-1 U-Net regressor — no lossy downsample in the trunk; residual-bicubic with a",
               "  zero-init head (step-0 = exact bicubic).  Biggest single lever (+0.38 dB).",
              "Orthogonal wavelet high-frequency loss — multi-scale L1 on Haar HL/LH/HH (λ≈8);",
              "  counters L1's mean-shrinkage of the high band at zero inference cost.",
              "Shift-ensemble of that loss (avg over dyadic offsets) ≈ shift-invariance — best perceptual arm.",
              "Muon optimizer (Newton–Schulz orthogonalised ≥2D updates) lr 5e-3 — beats AdamW at every lr.",
              "BSRGAN synthetic pretrain → RealSR finetune; D4 8-fold test-time augmentation."])

# ---- 5 ablation figure -----------------------------------------------------
image_slide("What actually moves RealSR ×2", "fig_ablation.png",
            caption="One change at a time, identical protocol. Mechanism levers (architecture + frequency loss) dominate; extra capacity saturates.")

# ---- 6 ceiling figure ------------------------------------------------------
image_slide("The empirical ceiling (training-free, no model)", "fig_ceiling.png",
            subtitle="diffusion/oracle.py — the three physical walls of RealSR",
            caption="Registration is the binding wall: at 0.3–0.4 px residual misalignment the ceiling is 34.4–35.7 dB — our 34.41 sits inside it. Corrects the “38–40 dB noise floor” folklore (noise is small here).")

# ---- 7 tradeoff figure -----------------------------------------------------
image_slide("Perception–distortion tradeoff, reproduced in-house", "fig_tradeoff.png",
            caption="Same test set & protocol: our regression is at the fidelity frontier; our own generative arm pays ~6 dB for no-reference IQA — the Blau–Michaeli law made concrete on RealSR.")

# ---- 8 baselines catalogue -------------------------------------------------
bullets_slide("Baselines: reproduced vs cited",
              ["Reproduced in-repo, SAME protocol (A-strict):",
               "  SwinIR (light/largeish) · EDSR-baseline ✓ · RCAN / SRResNet / RRDB (pipeline).",
              "Cited with an explicit protocol tag, never mixed into a claim:",
              "  GAN real-ISR: BSRGAN · Real-ESRGAN · LDL · FeMaSR  (D-blind / B-128/512).",
              "  Diffusion: StableSR · DiffBIR · SeeSR · ResShift · SinSR · OSEDiff · AdcSR ·",
              "    TSD-SR · TinySR · PURE · RealSR-R1  — 22.6–28.7 dB on a synthetically re-degraded",
              "    RealSR (a harder/different input).",
              "Full table + sources: docs/baseline/README.md  ·  citations: exp/RELATED_WORK_AND_UPPER_BOUND.md"])

# ---- 9 negative results ----------------------------------------------------
bullets_slide("Negative results — 28+ arms, all reported",
              ["Discriminative-capacity ceiling: capacity 5M→72.5M and pyramid flattening: saturating/negative.",
              "Mamba / VSS scan backbone: −2.3 dB at equal sample budget (memory-bound, poor sample efficiency).",
              "Loss/geometry regularisers tried & rejected: D4 equivariance, coord injection, freq-routing, gated FFN.",
              "Frequency experiments: bijective wavelet U-Net, dual-branch wavelet U-Net, DTCWT basis — none beat",
              "  the plain wavelet-HF loss + shift-ensemble.",
              "Generative: pixel/latent diffusion, frequency-domain rectified flow (HF residual), adversarial data",
              "  mining (N3) — fidelity collapses or plateau (consistent with the ceiling).",
              "Lesson: at 390 real pairs, discriminative regression already captures the recoverable signal."])

# ---- 10 protocol caveat ----------------------------------------------------
bullets_slide("Why protocol discipline matters",
              ["RealSR numbers are inconsistent across papers: Limited-range Y vs full-range RGB",
               "  (Y ≈ 0.4–0.8 dB lower), border shave, test subset, and Protocol A vs B.",
              "We report the strictest form and tag every external figure (A-strict / B-128-512 / C-REdeg / D-blind).",
              "Pretrained blind weights (Real-ESRGAN etc.) and diffusion SOTA are a different task on a",
              "  re-degraded input — comparing them to a paired-protocol PSNR would be misleading.",
              "Takeaway: our headline is a defensible same-protocol result, not a leaderboard number."])

# ---- 11 conclusions --------------------------------------------------------
bullets_slide("Conclusions & what's next",
              ["A recipe wins: stride-1 + wavelet-HF(+shift) + Muon + BSRGAN pretrain + TTA → 34.41 dB / 0.9285 SSIM.",
              "×2 PSNR is physics-capped (~34.4–35.7 dB, registration) — we are at the ceiling, not model-capped.",
              "Open fidelity headroom is at ×3/×4 (lighter-tuned here): 31.13 vs ceiling 34.4 (×3).",
              "Beyond fidelity: the frontier is perceptual (MUSIQ / LPIPS / FID) and data-limited at this scale.",
              "Reproduce:  README §9  ·  measure the ceiling: python -m diffusion.oracle --scale 2",
              "Artefacts: exp/FINAL_REPORT.md · docs/baseline/ · docs/figs · this deck (docs/slides.pptx)."])

# ---- 12 references ---------------------------------------------------------
bullets_slide("References",
              ["Cai et al., RealSR / LP-KPN, ICCV 2019.",
              "Liang et al., SwinIR, ICCV-W 2021 · Lim et al., EDSR, CVPR-W 2017 · Zhang et al., RCAN, ECCV 2018.",
              "Wang et al., ESRGAN, ECCV-W 2018 · Zhang et al., BSRGAN, ICCV 2021 · Wang et al., Real-ESRGAN, ICCV-W 2021.",
              "Qiao et al., RealSR-R1, arXiv:2506.16796 (2025) · TinySR, arXiv:2508.17434 (2025) — baseline tables.",
              "Blau & Michaeli, The Perception-Distortion Tradeoff, CVPR 2018 (DOI 10.1109/CVPR.2018.00652).",
              "Baker & Kanade, Limits on Super-Resolution..., IEEE TPAMI 2002 (DOI 10.1109/TPAMI.2002.1033210).",
              "Robinson & Milanfar, Fundamental Limits in Image Registration (TIP 2004) & Statistical SR (TIP 2006).",
              "Chatterjee & Milanfar, Is Denoising Dead?, IEEE TIP 2010 · Zhou et al., Updating the Evaluation of SR, CVPR 2021."])

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
